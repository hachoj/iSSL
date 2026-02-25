import math
import os
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

# DDP Code
from torch.nn.parallel import DistributedDataParallel as DDP

# Optimizers
from torch.optim import AdamW, Muon

import wandb
from core.utils import cosine_with_linear_wamrup
from scripts.utils.ddp_setup import init_ddp

torch.set_float32_matmul_precision("high")


def set_seed(base_seed: int, rank: int = 0) -> int:
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def loss_fn(
    pred: Float[Tensor, "B 1000"],
    label: Float[Tensor, "B"],
) -> torch.Tensor:
    return F.cross_entropy(pred, label)


def build_param_groups(model, lr, wd):
    muon_params = []  # 2D weights → Muon, with decay
    adamw_decay = []  # non-2D that still get decay (rare, but possible)
    adamw_no_decay = []  # biases, norms, embeddings → AdamW, no decay

    no_decay_keywords = {"bias", "norm", "ln_", "layernorm", "embedding"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Check if this param should skip weight decay
        skip_decay = any(kw in name.lower() for kw in no_decay_keywords)

        if p.ndim == 2 and not skip_decay:
            # Standard weight matrices → Muon
            muon_params.append(p)
        elif skip_decay:
            # Biases, norms, embeddings → AdamW without decay
            adamw_no_decay.append(p)
        else:
            # Everything else (e.g., 1D scales) → AdamW with decay
            adamw_decay.append(p)

    return {
        "muon": [{"params": muon_params, "lr": lr}],
        "adamw": [
            {"params": adamw_decay, "lr": lr, "weight_decay": wd},
            {"params": adamw_no_decay, "lr": lr, "weight_decay": 0.0},
        ],
    }


def strip_orig_mod(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def clean_key(key: str) -> str:
        cleaned = key.replace("_orig_mod", "")
        return cleaned.lstrip(".")

    return {clean_key(k): v for k, v in state_dict.items()}

def train(
    model,
    probe,
    adamw,
    muon,
    adamw_scheduler,
    muon_scheduler,
    train_dataloader,
    val_dataloader,
    cfg,
    is_main,
    device,
):
    if cfg.wandb.enabled and is_main:
        is_sweep = os.getenv("WANDB_SWEEP_ID") is not None
        if is_sweep:
            wandb.init(
                dir=cfg.wandb.dir,
            )
        else:
            wandb.init(
                project=cfg.wandb.project,
                # config={
                # },
                dir=cfg.wandb.dir,
                name=cfg.wandb.name,
            )
        wandb.define_metric("train_step")
        wandb.define_metric("epoch_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("epoch/*", step_metric="epoch_step")

    grad_clip = float("inf") if cfg.train.grad_clip < 0 else cfg.train.grad_clip

    optimizer_step = 0
    for epoch in range(cfg.train.num_epochs):
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_micro_steps = torch.zeros((), device=device)
        grad_norm = torch.zeros((), device=device)
        accum_window_loss = torch.zeros((), device=device)
        accum_window_steps = torch.zeros((), device=device)

        micro_step = 0
        for idx, batch in enumerate(train_dataloader):
            if micro_step == 0:
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)
                accum_window_loss = torch.zeros((), device=device)
                accum_window_steps = torch.zeros((), device=device)

            micro_step += 1
            image = batch[0].to(device=device, non_blocking=True)
            label = batch[1].to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    logits, _ = model(image)
                pred = probe(logits)
            pred = pred.to(dtype=torch.float32)
            loss = (
                loss_fn(pred, label)
                / cfg.train.grad_accum
            )
            accum_window_loss += loss.detach() * cfg.train.grad_accum
            accum_window_steps += 1
            epoch_loss_sum += loss.detach() * cfg.train.grad_accum
            epoch_micro_steps += 1
            if micro_step == cfg.train.grad_accum:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
                muon.step()
                adamw.step()
                muon_scheduler.step()
                adamw_scheduler.step()
                optimizer_step += 1
                if (
                    is_main
                    and cfg.wandb.enabled
                    and (
                        optimizer_step == 1
                        or optimizer_step % cfg.train.every_n_step == 0
                    )
                ):
                    wandb.log(
                        {
                            "train_step": optimizer_step,
                            "train/loss": (
                                accum_window_loss
                                / torch.clamp(accum_window_steps, min=1)
                            ).item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": adamw.param_groups[0]["lr"],
                        }
                    )
                micro_step = 0
                accum_window_steps = torch.zeros((), device=device)
            else:
                with probe.no_sync():
                    loss.backward()

        # --- Handle the case where the last few batches in the epoch don't fill up a grad_accum window ---
        if micro_step > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            muon.step()
            adamw.step()
            muon_scheduler.step()
            adamw_scheduler.step()
            optimizer_step += 1

        val_total_correct = torch.zeros((), device=device)
        val_total_preds = torch.zeros((), device=device)
        for idx, batch in enumerate(val_dataloader):
            image = batch[0].to(device=device, non_blocking=True)
            label = batch[1].to(device=device, non_blocking=True)
            B = label.shape[0]
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                logits, _ = model(image)
                pred = probe(logits)
            pred = pred.to(dtype=torch.float32)
            pred = pred.argmax(dim=1)
            val_total_correct += torch.sum(pred==label)
            val_total_preds += B
        val_total_correct_detached = val_total_correct.detach()
        val_total_preds_detached = val_total_preds.detach()
        dist.all_reduce(val_total_correct_detached, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_preds_detached, op=dist.ReduceOp.SUM)
        val_accuracy_detached = val_total_correct_detached / torch.clamp(
            val_total_preds_detached, min=1
        )

        loss_sum_detached = epoch_loss_sum.detach()
        loss_count_detached = epoch_micro_steps.detach()
        dist.all_reduce(loss_sum_detached, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count_detached, op=dist.ReduceOp.SUM)
        loss_detached = loss_sum_detached / torch.clamp(loss_count_detached, min=1)
        grad_norm_detached = grad_norm.detach()
        dist.all_reduce(grad_norm_detached, op=dist.ReduceOp.AVG)

        if is_main:
            if cfg.wandb.enabled:
                wandb.log(
                    {
                        "epoch_step": epoch + 1,
                        "epoch/train_loss": loss_detached.item(),
                        "epoch/grad_norm": grad_norm_detached.item(),
                        "epoch/val_top1": val_accuracy_detached.item(),  # pyrefly:ignore
                    }
                )
            save_dir = os.path.abspath(cfg.train.checkpoint_dir)
            run_id = wandb.run.id if wandb.run is not None else "no_wandb_run"
            save_dir = os.path.join(
                save_dir, f"{cfg.train.lr}_{cfg.train.weight_decay}_{run_id}"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_{epoch + 1}.pt")
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "probe": probe.module.state_dict(),
                    "optimizer_muon": muon.state_dict(),
                    "optimizer_adamw": adamw.state_dict(),
                    "step": epoch,
                },
                save_path,
            )


@hydra.main(version_base=None, config_path="../configs/probe_MAE", config_name="config")
def main(cfg: DictConfig):
    # setup DDP
    local_rank, rank, world_size = init_ddp()
    is_main: bool = rank == 0

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", init_method="env://"
    )  # nvidia collective communications library

    device = torch.device("cuda", local_rank)

    seed = set_seed(cfg.train.seed, rank=rank)
    if is_main:
        print(f"Base seed: {cfg.train.seed}; rank0 seed: {seed}")

    train_dataloader, val_dataloader = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    probe = hydra.utils.instantiate(cfg.probe)

    groups = build_param_groups(
        probe,
        lr=cfg.train.lr,
        wd=cfg.train.weight_decay,
    )

    muon = Muon(groups["muon"], adjust_lr_fn=cfg.train.muon_mode)
    adamw = AdamW(groups["adamw"], betas=tuple(cfg.train.adamw_betas))  # pyrefly:ignore

    optimizer_steps_per_epoch = max(
        1, math.ceil((1_281_167 // cfg.train.batch_size) / cfg.train.grad_accum)
    )
    muon_scheduler = cosine_with_linear_wamrup(
        optimizer=muon,
        num_epochs=cfg.train.num_epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )
    adamw_scheduler = cosine_with_linear_wamrup(
        optimizer=adamw,
        num_epochs=cfg.train.num_epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )

    step_start = 0
    if os.path.exists(cfg.train.checkpoint_init_dir):
        state_dict = torch.load(
            os.path.abspath(cfg.train.checkpoint_init_dir), map_location="cpu"
        )
        model.load_state_dict(strip_orig_mod(state_dict["model"]))
    else:
        raise ValueError("Missing model path")

    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Number of probe parameters: {sum(p.numel() for p in probe.parameters())}")

    # move to devices
    model = model.to(device=device)
    probe = probe.to(device=device)
    model = torch.compile(model)
    probe = torch.compile(probe)
    model = DDP(model, device_ids=[local_rank])
    probe = DDP(probe, device_ids=[local_rank])

    model.eval()

    train(
        model=model,
        probe=probe,
        adamw=adamw,
        muon=muon,
        adamw_scheduler=adamw_scheduler,
        muon_scheduler=muon_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        cfg=cfg,
        is_main=is_main,
        device=device,
    )

    dist.destroy_process_group()  # DDP cleanup


if __name__ == "__main__":
    main()
