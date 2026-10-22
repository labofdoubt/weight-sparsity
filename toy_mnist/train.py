"""Training loop of the toy study.

    python -m toy_mnist.train --config toy_mnist/configs/topk_dust.yaml train.run_name=x

Steps-based, as the main repository's trainer: AdamW with the warmup +
cosine schedule over ``train.max_steps``; under ``topk.surrogate_mode: dust``
the step kind follows ``dust_schedule`` and a DUST step runs
``toy_mnist.dust.dust_micro_step`` instead of the CE backward.  The run
directory ``train.out_dir/train.run_name`` receives ``config.json``,
``metrics.jsonl`` (one record per log window and per validation),
TensorBoard events (when the package is installed), ``summary.json`` at the
end and ``final.pt``.

Metrics: ``train/ce`` and ``train/acc`` are the clean hard-forward values on
both kinds of step; ``train/grad_norm`` is the last step's total gradient
norm, whichever kind; ``dust/*`` describe the DUST steps of the window only
(``dust/steps`` is 0 on a window without one); ``val/ce`` and ``val/acc`` are
measured on the 10 000 test images with the deterministic hard forward;
``gate/features_seen`` counts the input features active at least once in the
window's training forwards, ``gate/usage_eff`` is ``exp(H)`` of the normalized
usage EMA (the effective number of used features); ``rescale/g_*`` summarize
``g``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config, TrainConfig, load_config
from .data import MNISTData
from .dust import DustRunningStats, DustTrainingState, dust_micro_step
from .model import ToyMLP


# --------------------------------------------------------------------------- #
# optimizer and schedule (the main repository's conventions)
# --------------------------------------------------------------------------- #


def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay on the >= 2-D weights only (never on biases or g)."""
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    groups = [{"params": decay, "weight_decay": float(train_cfg.weight_decay), "name": "decay"},
              {"params": nodecay, "weight_decay": 0.0, "name": "nodecay"}]
    groups = [g for g in groups if g["params"]]
    if train_cfg.optimizer.lower() != "adamw":
        raise ValueError(f"unsupported optimizer: {train_cfg.optimizer}")
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=tuple(train_cfg.betas),
                             eps=train_cfg.eps)


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
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# evaluation and logging
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model: nn.Module, data: MNISTData, batch_size: int) -> Tuple[float, float]:
    """Mean CE and accuracy over the test set, deterministic hard forward."""
    was_training = model.training
    model.eval()
    ce_sum, correct, n = 0.0, 0, 0
    for x, y in data.val_batches(batch_size):
        logits, ce = model(x, y)
        ce_sum += float(ce) * x.shape[0]
        correct += int((logits.argmax(-1) == y).sum())
        n += x.shape[0]
    model.train(was_training)
    return ce_sum / max(1, n), correct / max(1, n)


class RunLogger:
    def __init__(self, run_dir: str, tensorboard: bool):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.jsonl = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(run_dir)
            except Exception as exc:  # tensorboard not installed
                print(f"[train] no TensorBoard writer ({exc}); metrics.jsonl only")

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        rec = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, float(v), step)

    def close(self) -> None:
        self.jsonl.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def gate_metrics(model: ToyMLP) -> Dict[str, float]:
    out: Dict[str, float] = {}
    gate = model.gate
    if gate is not None:
        with torch.no_grad():
            out["gate/features_seen"] = float(gate.seen.sum())
            gate.seen.zero_()
            usage = gate.usage.float()
            total = float(usage.sum())
            if total > 0:
                p = usage / total
                ent = -(p[p > 0] * p[p > 0].log()).sum()
                out["gate/usage_eff"] = float(ent.exp())
                out["gate/usage_max"] = float(usage.max())
    if model.rescale is not None:
        with torch.no_grad():
            g = model.rescale.log_scale.float()
            out.update({"rescale/g_mean": float(g.mean()), "rescale/g_std": float(g.std()),
                        "rescale/g_min": float(g.min()), "rescale/g_max": float(g.max())})
    return out


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def train(cfg: Config, data: Optional[MNISTData] = None,
          on_step: Optional[Callable[[int, nn.Module, torch.optim.Optimizer], None]] = None
          ) -> Dict[str, object]:
    """Train per ``cfg``; returns the summary (also written as summary.json).

    ``data`` lets a caller supply the tensors (tests use synthetic images);
    ``on_step(step, model, optimizer)`` is called at the start of every step,
    before the gradients are cleared.
    """
    tr = cfg.train
    device = resolve_device(tr.device)
    torch.manual_seed(tr.seed)
    if data is None:
        data = MNISTData.from_config(cfg.data, device, tr.seed)
    model = ToyMLP(cfg).to(device)
    optimizer = build_optimizer(model, tr)
    params = [p for p in model.parameters() if p.requires_grad]

    dust: Optional[DustTrainingState] = None
    dust_optimizer: Optional[torch.optim.Optimizer] = None
    dust_stats = DustRunningStats()
    if cfg.topk.enabled and cfg.topk.surrogate_mode == "dust":
        dust = DustTrainingState(cfg.topk, device, tr.seed)
        print(f"[train] dust: {dust.describe()}; train/ce is the clean hard-forward CE on both "
              "kinds of step, dust/* describe the DUST steps only")
        if cfg.topk.dust_optimizer == "separate":
            dust_optimizer = build_optimizer(model, tr)
            print("[train] dust: separate optimizer for the DUST steps (own Adam moments and "
                  "step count)")

    run_dir = os.path.join(tr.out_dir, tr.run_name)
    logger = RunLogger(run_dir, tr.tensorboard)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[train] {run_dir}: {model}")
    print(f"[train] parameters: {sum(p.numel() for p in params)}; device {device}")

    best_val_ce, best_val_acc, best_step = float("inf"), 0.0, 0
    last_val: Optional[Tuple[float, float]] = None
    val_history = []
    status = "finished"
    running_ce = running_acc = 0.0
    running_n = 0
    grad_norm = torch.tensor(0.0)
    t0 = time.time()
    t_start = t0
    step1 = 0

    def validate(step1: int) -> None:
        nonlocal best_val_ce, best_val_acc, best_step, last_val
        ce, acc = evaluate(model, data, tr.val_batch_size)
        last_val = (ce, acc)
        val_history.append({"step": step1, "val/ce": ce, "val/acc": acc})
        if ce < best_val_ce:
            best_val_ce, best_val_acc, best_step = ce, acc, step1
        logger.log(step1, {"val/ce": ce, "val/acc": acc})
        print(f"[val] step {step1} ce {ce:.4f} acc {acc:.4f} (best ce {best_val_ce:.4f} "
              f"at {best_step})")

    for step in range(tr.max_steps):
        lr = lr_at(step, tr)
        set_lr(optimizer, lr)
        dust_step = dust is not None and dust.kind(step) == "dust"
        if dust_optimizer is not None:
            set_lr(dust_optimizer, lr * dust.lr_scale)
        elif dust_step:
            set_lr(optimizer, lr * dust.lr_scale)  # this step only; reset above next step
        if on_step is not None:
            on_step(step, model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        if dust_optimizer is not None:
            dust_optimizer.zero_grad(set_to_none=True)
        step1 = step + 1
        internal_stop: Optional[str] = None

        x, y = data.train_batch(tr.batch_size)
        if dust_step:
            st = dust_micro_step(model, x, y, dust, want_cosine=dust.cosine_due())
            dust.dust_steps += 1
            ce_val, acc_val = st["ce_clean"], st["acc_clean"]
            dust_stats.add(st)
            if not math.isfinite(ce_val):
                internal_stop = f"non-finite training CE at step {step1} (dust step)"
        else:
            model.train()
            logits, ce = model(x, y)
            ce.backward()
            ce_val = float(ce.detach())
            acc_val = float((logits.detach().argmax(-1) == y).float().mean())
            if not math.isfinite(ce_val):
                internal_stop = f"non-finite training CE at step {step1}"

        active_opt = dust_optimizer if (dust_step and dust_optimizer is not None) else optimizer
        max_norm = tr.grad_clip if tr.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if internal_stop is None and not math.isfinite(float(grad_norm)):
            internal_stop = f"non-finite gradient norm at step {step1}"
        if internal_stop is None:
            active_opt.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            status = f"stopped: {internal_stop}"
            print(f"[train] {status}")
        if dust_step:
            dust_stats.add_step(float(grad_norm))

        running_ce += ce_val
        running_acc += acc_val
        running_n += 1

        if step1 % tr.log_every_steps == 0 or step1 == 1 or internal_stop is not None:
            dt = time.time() - t0
            t0 = time.time()
            metrics = {
                "train/ce": running_ce / running_n,
                "train/acc": running_acc / running_n,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
                "perf/ms_per_step": 1000 * dt / running_n,
                "perf/epoch": float(data.epoch),
            }
            metrics.update(gate_metrics(model))
            dm: Dict[str, float] = {}
            if dust is not None:
                dm = dust_stats.metrics()
                dust_stats.reset()
                metrics.update(dm)
                metrics["dust/lr"] = lr * dust.lr_scale
            logger.log(step1, metrics)
            line = (f"[train] step {step1} ce {metrics['train/ce']:.4f} acc "
                    f"{metrics['train/acc']:.4f} lr {lr:.2e} gnorm {float(grad_norm):.3e} "
                    f"{metrics['perf/ms_per_step']:.1f} ms/step")
            if "gate/features_seen" in metrics:
                line += f" | seen {int(metrics['gate/features_seen'])}"
            if dm.get("dust/steps"):
                line += (f" | dust {int(dm['dust/steps'])} gap {dm['dust/ce_gap']:+.4f}"
                         f" ghat {dm['dust/ghat_rms']:.2e}")
                if "dust/cosine" in dm:
                    line += f" cos {dm['dust/cosine']:.3f}"
            print(line)
            running_ce = running_acc = 0.0
            running_n = 0

        if internal_stop is not None:
            break
        if tr.validate_every_steps > 0 and step1 % tr.validate_every_steps == 0 \
                and step1 != tr.max_steps:
            validate(step1)

    validate(step1)
    train_time = time.time() - t_start
    summary = {
        "run_dir": run_dir,
        "status": status,
        "steps": step1,
        "dust_steps": dust.dust_steps if dust is not None else 0,
        "best_val_ce": best_val_ce,
        "best_val_acc": best_val_acc,
        "best_step": best_step,
        "final_val_ce": last_val[0] if last_val else None,
        "final_val_acc": last_val[1] if last_val else None,
        "train_time_s": train_time,
        "epochs": data.epoch,
        "val_history": val_history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if tr.save_checkpoint:
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "config": cfg.to_dict(), "step": step1}
        if dust is not None:
            payload["dust_state"] = dust.state_dict()
        if dust_optimizer is not None:
            payload["dust_optimizer"] = dust_optimizer.state_dict()
        torch.save(payload, os.path.join(run_dir, "final.pt"))
    logger.close()
    print(f"[train] {status}; {step1} steps in {train_time:.1f} s; best val ce "
          f"{best_val_ce:.4f} acc {best_val_acc:.4f} at step {best_step}")
    return summary


def main(argv=None) -> Dict[str, object]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="YAML config (with optional _base_)")
    ap.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    return train(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
