"""Small helpers: device/dtype resolution, seeding, logging."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in dtypes:
        raise ValueError(f"unknown dtype: {name}")
    dtype = dtypes[name]
    if device.type == "cpu" and dtype is torch.float16:
        return torch.float32
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return dtype


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and dtype is torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return _NullContext()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.1f}T"


class Logger:
    """Console + JSONL (+ optional Weights & Biases) logging."""

    def __init__(
        self,
        out_dir: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        wandb_project: str = "",
        wandb_entity: str = "",
    ):
        self.dir = os.path.join(out_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "metrics.jsonl")
        self.start = time.time()
        self.wandb = None
        if wandb_project:
            try:
                import wandb

                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity or None,
                    name=run_name,
                    config=config,
                )
                self.wandb = wandb
            except Exception as exc:  # pragma: no cover
                print(f"[log] wandb disabled ({exc})")
        if config is not None:
            with open(os.path.join(self.dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    def log(self, step: int, metrics: Dict[str, float], console: str = "") -> None:
        record = {"step": step, "wall_s": round(time.time() - self.start, 2), **metrics}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if console:
            print(console, flush=True)

    def close(self) -> None:
        if self.wandb is not None:  # pragma: no cover
            self.wandb.finish()
