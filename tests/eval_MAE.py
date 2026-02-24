import math
import os
import random
import time
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from jaxtyping import Bool, Float
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

# DDP Code
from torch.nn.parallel import DistributedDataParallel as DDP

# Optimizers
from torch.optim import AdamW, Muon

from core.utils import cosine_with_linear_wamrup

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
    image: Float[Tensor, "B 3 H W"],
    pred: Float[Tensor, "B 3 H W"],
    mask: Bool[Tensor, "B N"],
    patch_stride: Tuple[int, int],
    normalize_per_patch: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute MSE loss on masked patches with optional per-patch normalization.
    mask: True for masked patches, False for visible patches.
    Loss is computed only on masked patches.
    """
    B, _, H, W = image.shape
    p1, p2 = patch_stride
    h, w = H // p1, W // p2

    # Convert images to patches: B C H W -> B N (C*p1*p2)
    image_patches: Float[Tensor, "B N d"] = rearrange(
        image, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", h=h, w=w, p1=p1, p2=p2
    )
    pred_patches: Float[Tensor, "B N d"] = rearrange(
        pred, "B C (h p1) (w p2) -> B (h w) (C p1 p2)", h=h, w=w, p1=p1, p2=p2
    )

    if normalize_per_patch:
        # Normalize each patch by its own mean and std (as in MAE paper)
        mean: Float[Tensor, "B N 1"] = image_patches.mean(dim=-1, keepdim=True)
        var: Float[Tensor, "B N 1"] = image_patches.var(dim=-1, keepdim=True)
        image_patches = (image_patches - mean) / (var + eps).sqrt()

    # Compute MSE per patch
    loss_per_patch: Float[Tensor, "B N"] = (
        (pred_patches - image_patches).pow(2).mean(dim=-1)
    )

    # Average loss over all masked patches
    num_masked = mask.sum()
    if num_masked > 0:
        loss = (loss_per_patch * mask.to(loss_per_patch.dtype)).sum() / num_masked
    else:
        loss = torch.tensor(0.0, device=image.device, dtype=image.dtype)

    return loss


def build_param_groups(model, lr_adamw, lr_muon, wd):
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
        "muon": [{"params": muon_params, "lr": lr_muon}],
        "adamw": [
            {"params": adamw_decay, "lr": lr_adamw, "weight_decay": wd},
            {"params": adamw_no_decay, "lr": lr_adamw, "weight_decay": 0.0},
        ],
    }


def strip_orig_mod(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def clean_key(key: str) -> str:
        cleaned = key.replace("_orig_mod", "")
        return cleaned.lstrip(".")

    return {clean_key(k): v for k, v in state_dict.items()}


def make_wandb_recon_pair(
    pred: torch.Tensor,
    image: torch.Tensor,
) -> wandb.Image:
    # Undo ImageNet normalization from data.py and place pred/target side-by-side.
    mean = pred.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pred.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    pred_unnorm = (pred[:1] * std + mean).clamp(0.0, 1.0)
    image_unnorm = (image[:1] * std + mean).clamp(0.0, 1.0)
    pair = torch.cat((pred_unnorm, image_unnorm), dim=-1)
    return wandb.Image(pair[0].detach().cpu(), caption="pred | image")


def train(
    model,
    decoder,
    adamw,
    muon,
    adamw_scheduler,
    muon_scheduler,
    train_dataloader,
    val_dataloader,
    cfg,
    is_main,
    device,
    step_start=0,
):
    if cfg.wandb.enabled and is_main:
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

    start_time = time.time()

    optimizer_step = step_start
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
            # -- image net label is not needed for MAE training --
            # label = batch[1].to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, mask = model(image, mask_ratio=cfg.train.mask_ratio)
                pred = decoder(logits, mask)
            pred = pred.to(dtype=torch.float32)
            loss = (
                loss_fn(image, pred, mask, patch_stride=decoder.module.patch_stride)
                / cfg.train.grad_accum
            )
            accum_window_loss += loss.detach() * cfg.train.grad_accum
            accum_window_steps += 1
            epoch_loss_sum += loss.detach() * cfg.train.grad_accum
            epoch_micro_steps += 1
            if micro_step == cfg.train.grad_accum:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(decoder.parameters()), grad_clip
                )
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
                with model.no_sync(), decoder.no_sync():
                    loss.backward()

        # --- Handle the case where the last few batches in the epoch don't fill up a grad_accum window ---
        if micro_step > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoder.parameters()), grad_clip
            )
            muon.step()
            adamw.step()
            muon_scheduler.step()
            adamw_scheduler.step()
            optimizer_step += 1

        if epoch == 0 or (epoch + 1) % cfg.train.every_n_val == 0:
            val_loss_sum = torch.zeros((), device=device)
            val_steps = torch.zeros((), device=device)
            for idx, batch in enumerate(val_dataloader):
                image = batch[0].to(device=device, non_blocking=True)
                with (
                    torch.no_grad(),
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
                ):
                    logits, mask = model(image, mask_ratio=cfg.train.mask_ratio)
                    pred = decoder(logits, mask)
                pred = pred.to(dtype=torch.float32)
                val_loss = loss_fn(
                    image, pred, mask, patch_stride=decoder.module.patch_stride
                )
                val_loss_sum += val_loss.detach()
                val_steps += 1
            val_sum_detached = val_loss_sum.detach()
            val_count_detached = val_steps.detach()
            dist.all_reduce(val_sum_detached, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_detached, op=dist.ReduceOp.SUM)
            val_loss_detached = val_sum_detached / torch.clamp(
                val_count_detached, min=1
            )

        if cfg.wandb.enabled and (epoch + 1) % cfg.train.every_n_epoch == 0:
            loss_sum_detached = epoch_loss_sum.detach()
            loss_count_detached = epoch_micro_steps.detach()
            dist.all_reduce(loss_sum_detached, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count_detached, op=dist.ReduceOp.SUM)
            loss_detached = loss_sum_detached / torch.clamp(loss_count_detached, min=1)
            grad_norm_detached = grad_norm.detach()
            dist.all_reduce(grad_norm_detached, op=dist.ReduceOp.AVG)
            if is_main and (epoch == 0 or (epoch + 1) % cfg.train.every_n_val == 0):
                wandb.log(
                    {
                        "epoch_step": epoch + 1,
                        "epoch/train_loss": loss_detached.item(),
                        "epoch/grad_norm": grad_norm_detached.item(),
                        "epoch/val_loss": val_loss_detached.item(),  # pyrefly:ignore
                        "epoch/pred_vs_image": make_wandb_recon_pair(
                            pred, image  # pyrefly:ignore
                        ),
                    }
                )
            elif is_main:
                wandb.log(
                    {
                        "epoch_step": epoch + 1,
                        "epoch/train_loss": loss_detached.item(),
                        "epoch/grad_norm": grad_norm_detached.item(),
                    }
                )
        if (epoch + 1) % cfg.train.every_n_checkpoint == 0 and is_main:
            save_dir = os.path.abspath(cfg.train.checkpoint_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_{epoch + 1}.pt")
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "decoder": decoder.module.state_dict(),
                    "optimizer_muon": muon.state_dict(),
                    "optimizer_adamw": adamw.state_dict(),
                    "step": epoch,
                },
                save_path,
            )


@hydra.main(version_base=None, config_path="../configs/MAE", config_name="config")
def main(cfg: DictConfig):
    # setup DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # world_size and rank are redundant on one node but good practice
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
    decoder = hydra.utils.instantiate(cfg.decoder)

    groups = build_param_groups(
        nn.ModuleDict({"encoder": model, "decoder": decoder}),
        lr_adamw=cfg.train.lr_adamw,
        lr_muon=cfg.train.lr_muon,
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
    if os.path.exists(cfg.train.checkpoint_init_dir) and cfg.train.is_restore:
        state_dict = torch.load(
            os.path.abspath(cfg.train.checkpoint_init_dir), map_location="cpu"
        )
        model.load_state_dict(strip_orig_mod(state_dict["model"]))
        if "decoder" in state_dict:
            decoder.load_state_dict(strip_orig_mod(state_dict["decoder"]))

    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # move to devices
    model = model.to(device=device)
    decoder = decoder.to(device=device)
    model = torch.compile(model)
    # decoder = torch.compile(decoder)
    model = DDP(model, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])

    eval(
        model=model,
        decoder=decoder,
        adamw=adamw,
        muon=muon,
        adamw_scheduler=adamw_scheduler,
        muon_scheduler=muon_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        cfg=cfg,
        is_main=is_main,
        device=device,
        step_start=step_start,
    )

    dist.destroy_process_group()  # DDP cleanup


if __name__ == "__main__":
    main()
