"""Optimizer construction and the learning-rate schedule.

Three parameter groups:

1. ``decay``  -- every >= 2D model weight (matmuls, embeddings): weight decay on.
2. ``nodecay`` -- 1D/0D parameters (RMSNorm gains, any biases): weight decay off.
Only the first two groups exist since the weight-sparsity removal.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .config import TrainConfig


def build_optimizer(
    model,
    train_cfg: TrainConfig,
):
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            nodecay.append(p)
    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay, "name": "decay", "is_mask": False},
        {"params": nodecay, "weight_decay": 0.0, "name": "nodecay", "is_mask": False},
    ]

    groups = [g for g in groups if g["params"]]
    if train_cfg.optimizer.lower() != "adamw":
        raise ValueError(f"unsupported optimizer: {train_cfg.optimizer} (only adamw for now)")
    optimizer = torch.optim.AdamW(
        groups,
        lr=train_cfg.lr,
        betas=tuple(train_cfg.betas),
        eps=train_cfg.eps,
    )
    return optimizer


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Warmup + (cosine | linear | constant) decay to ``min_lr_ratio * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if cfg.lr_schedule == "constant":
        return cfg.lr
    total = max(1, cfg.max_steps - cfg.warmup_steps)
    p = min(1.0, (step - cfg.warmup_steps) / total)
    min_lr = cfg.lr * cfg.min_lr_ratio
    if cfg.lr_schedule == "linear":
        return cfg.lr + (min_lr - cfg.lr) * p
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * p))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply ``lr`` to every non-mask group.

    A mask group keeps its own fixed lr, unless it was built with an
    ``lr_mult``, in which case it tracks ``lr_mult * lr``.
    """
    for group in optimizer.param_groups:
        if group.get("lr_mult") is not None:
            group["lr"] = lr * group["lr_mult"]
        elif not group.get("is_mask", False):
            group["lr"] = lr


def count_parameter_groups(optimizer: torch.optim.Optimizer) -> Dict[str, int]:
    return {
        g.get("name", str(i)): sum(p.numel() for p in g["params"])
        for i, g in enumerate(optimizer.param_groups)
    }
