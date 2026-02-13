import copy
import math
import os
import random
import re
import time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

# DDP Code
from torch.nn.parallel import DistributedDataParallel as DDP

# Optimizers
from torch.optim import AdamW, Muon

import wandb
from hiera.helpers import CosineWithLinearWarmup


def set_seed(base_seed: int, rank: int = 0) -> int:
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> CosineWithLinearWarmup:
    for group in optimizer.param_groups:
        group["lr"] = lr
    return CosineWithLinearWarmup(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
    )


def loss_fn(
    image: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    normalize_per_patch: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if image.shape != pred.shape:
        raise ValueError(
            f"Expected image and pred to have the same shape, got {image.shape} vs {pred.shape}."
        )
    if mask.ndim != 2 or mask.shape[0] != image.shape[0]:
        raise ValueError(
            f"Expected mask shape [B, num_windows], got {mask.shape} for batch size {image.shape[0]}."
        )

    b, c, h, w = image.shape
    num_windows = mask.shape[1]
    grid = math.isqrt(num_windows)
    if grid * grid != num_windows:
        raise ValueError(f"num_windows={num_windows} is not a perfect square.")
    if h % grid != 0 or w % grid != 0:
        raise ValueError(
            f"Image spatial shape {(h, w)} is not divisible by window grid {grid}x{grid}."
        )

    patch_h = h // grid
    patch_w = w // grid

    image_patches = rearrange(
        image,
        "B C (GH PH) (GW PW) -> B (GH GW) (C PH PW)",
        GH=grid,
        GW=grid,
        PH=patch_h,
        PW=patch_w,
    )
    pred_patches = rearrange(
        pred,
        "B C (GH PH) (GW PW) -> B (GH GW) (C PH PW)",
        GH=grid,
        GW=grid,
        PH=patch_h,
        PW=patch_w,
    )

    if normalize_per_patch:
        target_mean = image_patches.mean(dim=-1, keepdim=True)
        target_var = image_patches.var(dim=-1, keepdim=True, unbiased=False)
        image_patches = (image_patches - target_mean) / torch.sqrt(target_var + eps)

    per_patch_mse = (pred_patches - image_patches).pow(2).mean(dim=-1)
    masked_windows = ~mask.bool()
    denom = masked_windows.sum().clamp_min(1)
    return (per_patch_mse * masked_windows).sum() / denom


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
            name=cfg.wandb.name,
        )
        wandb.define_metric("train_step")
        wandb.define_metric("epoch_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("epoch/*", step_metric="epoch_step")

    grad_clip = float("inf") if cfg.train.grad_clip < 0 else cfg.train.grad_clip
    train_steps_per_epoch = max(1, 1_281_167 // cfg.train.batch_size)
    val_steps_per_epoch = max(1, 50_000 // cfg.train.batch_size)

    start_time = time.time()

    optimizer_step = step_start
    for epoch in range(cfg.train.num_epochs):
        epoch_loss = torch.zeros((), device=device)
        grad_norm = torch.zeros((), device=device)
        accum_window_loss = torch.zeros((), device=device)

        micro_step = 0
        for idx, batch in enumerate(train_dataloader):
            if micro_step == 0:
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)
                accum_window_loss = torch.zeros((), device=device)
            micro_step += 1
            image = batch[0].to(device=device, non_blocking=True)
            # -- image net label is not needed for MAE training --
            # label = batch[1].to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, mask = model(image, mask_ratio=cfg.train.mask_ratio)
                pred = decoder(logits, mask)
            pred = pred.to(dtype=torch.float32)
            loss = loss_fn(image, pred, mask) / cfg.train.grad_accum
            accum_window_loss += loss.detach() * cfg.train.grad_accum
            epoch_loss += (loss.detach() * cfg.train.grad_accum) / train_steps_per_epoch
            if micro_step == cfg.train.grad_accum:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(decoder.parameters()), grad_clip
                )
                muon.step()
                adamw.step()
                muon_scheduler.step_update()
                adamw_scheduler.step_update()
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
                                accum_window_loss / cfg.train.grad_accum
                            ).item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": adamw.param_groups[0]["lr"],
                            "train/pred_vs_image": make_wandb_recon_pair(pred, image),
                        }
                    )
                micro_step = 0
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
            muon_scheduler.step_update()
            adamw_scheduler.step_update()
            optimizer_step += 1

        total_val_loss = torch.zeros((), device=device)
        for idx, batch in enumerate(val_dataloader):
            image = batch[0].to(device=device, non_blocking=True)
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                logits, mask = model(image, mask_ratio=cfg.train.mask_ratio)
                pred = decoder(logits, mask)
            pred = pred.to(dtype=torch.float32)
            val_loss = loss_fn(image, pred, mask)
            total_val_loss += val_loss.detach() / val_steps_per_epoch

        if cfg.wandb.enabled and (epoch + 1) % cfg.train.every_n_epoch == 0:
            loss_detached = epoch_loss.detach()
            val_loss_detached = total_val_loss.detach()
            dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_detached, op=dist.ReduceOp.AVG)
            grad_norm_detached = grad_norm.detach()
            dist.all_reduce(grad_norm_detached, op=dist.ReduceOp.AVG)
            if is_main:
                wandb.log(
                    {
                        "epoch_step": epoch + 1,
                        "epoch/train_loss": loss_detached.item(),
                        "epoch/grad_norm": grad_norm_detached.item(),
                        "epoch/val_loss": val_loss_detached.item(),
                        "epoch/pred_vs_image": make_wandb_recon_pair(pred, image),
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
                    "step": epoch,
                },
                save_path,
            )


@hydra.main(version_base=None, config_path="config", config_name="config")
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
    adamw = AdamW(groups["adamw"], betas=tuple(cfg.train.adamw_betas))
    optimizer_steps_per_epoch = max(
        1, math.ceil((1_281_167 // cfg.train.batch_size) / cfg.train.grad_accum)
    )
    muon_scheduler = build_scheduler(
        optimizer=muon,
        lr=cfg.train.lr_muon,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.num_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )
    adamw_scheduler = build_scheduler(
        optimizer=adamw,
        lr=cfg.train.lr_adamw,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.num_epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
    )

    # TODO, implement resume logic
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
    # model = torch.compile(model)
    # decoder = torch.compile(decoder)
    model = DDP(model, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])

    train(
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
