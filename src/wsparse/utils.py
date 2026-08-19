"""Small helpers: device/dtype resolution, seeding, logging."""

from __future__ import annotations

import html
import json
import math
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
    """Resolve the requested dtype against what the device actually supports.

    Both fallbacks are announced rather than applied silently: they change the
    numerics of the run, and the bf16 -> fp16 one in particular also switches
    gradient loss scaling on (see the GradScaler in ``train``), which is easy to
    miss when comparing runs across machines.
    """
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in dtypes:
        raise ValueError(f"unknown dtype: {name}")
    dtype = dtypes[name]
    if device.type == "cpu" and dtype is torch.float16:
        print("[dtype] float16 is not supported on cpu -- falling back to float32")
        return torch.float32
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        name = torch.cuda.get_device_name(0)
        print(
            f"[dtype] bfloat16 is not supported by this GPU ({name}) -- falling back to "
            "float16, which also enables gradient loss scaling. Pass "
            "--train.dtype=float32 to avoid mixed precision entirely."
        )
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
    """Console + JSONL (+ optional TensorBoard / Weights & Biases) logging.

    Metric keys are already namespaced with ``/`` (``train/ce``, ``val/ce``,
    ``sparsity/...``, ``bottleneck/...``), which is exactly TensorBoard's
    grouping convention, so every panel lands in the right section for free.
    Events go to ``<out_dir>/<run_name>/tb``; pointing TensorBoard at
    ``<out_dir>`` therefore overlays every run in one chart, which is how the
    setups are meant to be compared.
    """

    def __init__(
        self,
        out_dir: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        wandb_project: str = "",
        wandb_entity: str = "",
        tensorboard: bool = False,
    ):
        self.dir = os.path.join(out_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "metrics.jsonl")
        self.samples_path = os.path.join(self.dir, "samples.txt")
        self.start = time.time()
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(
                    log_dir=os.path.join(self.dir, "tb"), flush_secs=30
                )
                if config is not None:
                    self.tb.add_text(
                        "config", "```json\n" + json.dumps(config, indent=2) + "\n```", 0
                    )
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[log] tensorboard disabled ({exc})")
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
        if self.tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self.tb.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if console:
            print(console, flush=True)

    def log_text(self, step: int, tag: str, text: str) -> None:
        """Free text (generated samples) to samples.txt, TensorBoard and wandb.

        Kept out of ``log`` because ``metrics.jsonl`` and the TensorBoard scalar
        path are numeric-only.
        """
        with open(self.samples_path, "a") as f:
            f.write(f"=== step {step} [{tag}]\n{text}\n\n")
        if self.tb is not None:
            self.tb.add_text(tag, text, step)
        if self.wandb is not None:  # pragma: no cover - optional dependency
            try:
                self.wandb.log(
                    {tag: self.wandb.Html(f"<pre>{html.escape(text)}</pre>")}, step=step
                )
            except Exception:
                pass

    def log_histogram(self, step: int, tag: str, values) -> None:
        """A distribution (TensorBoard HISTOGRAMS / DISTRIBUTIONS tabs)."""
        if self.tb is not None:
            self.tb.add_histogram(tag, values, step)

    def log_figure(self, step: int, tag: str, figure) -> None:
        """A rendered matplotlib figure (TensorBoard IMAGES tab, step slider)."""
        if self.tb is not None:
            self.tb.add_figure(tag, figure, step, close=True)
        elif figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb is not None:  # pragma: no cover
            self.wandb.finish()
