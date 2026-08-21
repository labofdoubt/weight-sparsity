"""Training loop.

    python -m wsparse.train --config configs/ltp_base.yaml [--train.lr=6e-4 ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from .config import Config, config_from_dict, load_config
from .data import build_streams, load_meta
from .model import build_model
from .optim import build_optimizer, count_parameter_groups, lr_at, set_lr
from .bottleneck import ActivationBottleneckController, apply_activation_bottleneck
from .sparsity import SparsityController, apply_sparsity
from .utils import Logger, autocast_context, human, resolve_device, resolve_dtype, set_seed


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(
    model,
    stream,
    batch_size: int,
    batches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    stride = batch_size * (stream.seq_len + 1)  # disjoint windows across batches
    for i in range(batches):
        x, y = stream.batch(batch_size, device, deterministic_offset=i * stride)
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train(was_training)
    ce = sum(losses) / max(1, len(losses))
    return {"ce": ce, "ppl": math.exp(min(20.0, ce))}


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: str, cfg: Config, model, optimizer, step: int, extra: Optional[Dict] = None
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_old_checkpoints(run_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "ckpt_step*.pt")),
        key=lambda p: int(os.path.basename(p).split("step")[1].split(".")[0]),
    )
    for old in ckpts[:-keep]:
        os.remove(old)


def load_for_inference(path: str, device: str = "cpu") -> Tuple[torch.nn.Module, Config, SparsityController]:
    """Rebuild a model (+ sparsity wrappers) from a checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(payload["config"])
    model = build_model(cfg.model)
    controller = apply_sparsity(model, cfg.sparsity, max_steps=cfg.train.max_steps)
    apply_activation_bottleneck(
        model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    controller.set_step(payload.get("step", cfg.train.max_steps))
    return model, cfg, controller


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train(cfg: Config) -> Dict[str, float]:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = load_meta(cfg.data.data_dir)
    cfg.model.vocab_size = int(meta["vocab_size"])
    train_stream, val_stream = build_streams(cfg.data, seed=cfg.train.seed)

    model = build_model(cfg.model).to(device)
    controller = apply_sparsity(model, cfg.sparsity, max_steps=cfg.train.max_steps)
    bottleneck = apply_activation_bottleneck(
        model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps
    )
    model.to(device)  # sparsity / bottleneck parameters created on cpu -> move again

    optimizer = build_optimizer(
        model, cfg.train, cfg.sparsity, mask_param_ids=controller.mask_parameter_ids()
    )
    # Sparsity parameters are clipped separately: dL/dtau sums over every weight
    # in the layer, so a shared global norm would let it squash the weight
    # gradients (LTP section 4.1 makes the same point about its magnitude).
    # sparsity.mask_grad_clip gives them their own threshold, which matters at
    # large beta -- dL/ds carries a factor beta*p*(1-p).
    mask_params = controller.mask_parameters()
    mask_ids = controller.mask_parameter_ids()
    weight_params = [p for p in model.parameters() if id(p) not in mask_ids]
    mask_clip = cfg.sparsity.mask_grad_clip
    mask_clip = cfg.train.grad_clip if mask_clip is None else float(mask_clip)

    run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    logger = Logger(
        cfg.train.out_dir,
        cfg.train.run_name,
        config=cfg.to_dict(),
        wandb_project=cfg.train.wandb_project,
        wandb_entity=cfg.train.wandb_entity,
        tensorboard=cfg.train.tensorboard,
    )
    cfg.dump(os.path.join(run_dir, "config.yaml"))

    start_step = 0
    resume_path = cfg.train.resume
    if resume_path == "auto":
        resume_path = os.path.join(run_dir, "latest.pt")
        resume_path = resume_path if os.path.exists(resume_path) else ""
    if resume_path:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        print(f"[train] resumed from {resume_path} at step {start_step}")

    if cfg.train.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype is torch.float16 and device.type == "cuda"))
    accum = cfg.train.grad_accum_steps
    micro_bs = int(cfg.train.micro_batch_size)
    tokens_per_step = cfg.train.batch_size * cfg.data.seq_len

    print(
        f"[train] device={device} dtype={dtype} params={human(model_params(model))} "
        f"(non-emb {human(model_params(model, non_embedding=True))}) "
        f"batch={cfg.train.batch_size}x{cfg.data.seq_len} tok "
        f"(micro {micro_bs} x accum {accum})"
    )
    print(f"[train] param groups: {count_parameter_groups(optimizer)}")
    if controller.enabled:
        print(
            f"[train] sparsity: method={cfg.sparsity.method} targets={cfg.sparsity.targets} "
            f"layers={len(controller.layers)} maskable={human(controller.total_maskable)} "
            f"beta {cfg.sparsity.beta_start:g} -> {cfg.sparsity.beta_end:g} "
            f"({cfg.sparsity.beta_schedule})"
        )
        if cfg.sparsity.method == "topk":
            first = controller.layers[0][1]
            print(
                f"[train] topk: k={first.k} j={first.j} per group "
                f"(groups={cfg.sparsity.topk_groups}), forward density "
                f"{controller.stats()['sparsity/density_topk']:.4f}, "
                f"w grad on {'topk+j' if first.w_grad_explore else 'topk'}"
            )

    if bottleneck.enabled and cfg.activation_bottleneck.calibrate_output and start_step == 0:
        # Before the first step only: on resume the fitted scale comes back with
        # the checkpoint, and re-fitting it against a trained model would be a
        # silent change of function mid-run.
        cal = bottleneck.calibrate_output_scale(
            lambda: train_stream.batch(micro_bs, device)[0],
            batches=cfg.activation_bottleneck.calibration_batches,
            iters=cfg.activation_bottleneck.calibration_iters,
        )
        print(
            f"[train] bottleneck output calibrated: scale mean "
            f"{cal['bottleneck/output_scale']:.4f} "
            f"(min {cal['bottleneck/output_scale_min']:.4f}, "
            f"max {cal['bottleneck/output_scale_max']:.4f})"
        )

    if bottleneck.enabled:
        cb = cfg.activation_bottleneck
        print(
            f"[train] activation bottleneck: {len(bottleneck.layers)} layers "
            f"({cfg.activation_bottleneck.layers}) {cb.placement} "
            f"N={cb.n_features} K={cb.k} J={cb.j} n_eff={cb.n_eff:g} "
            f"({cb.selection_mode}, {cb.boundary_mode}, {cb.effective_count_metric}, "
            f"{cb.surrogate_mode}) density={cb.k / cb.n_features:.3f} "
            f"params={human(bottleneck.n_parameters)}"
        )

    model.train()
    t0 = time.time()
    running_ce, running_n = 0.0, 0
    best_val = float("inf")
    last_metrics: Dict[str, float] = {}

    for step in range(start_step, cfg.train.max_steps):
        lr = lr_at(step, cfg.train)
        set_lr(optimizer, lr)
        controller.set_step(step)
        bottleneck.set_step(step)

        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        for _ in range(accum):
            x, y = train_stream.batch(micro_bs, device)
            with autocast_context(device, dtype):
                _, ce = model(x, y)
            ce_sum += ce.detach().float().item()
            scaler.scale(ce / accum).backward()

        # sparsity penalty: added once per optimiser step (it does not depend on
        # the batch), so its gradient is not divided by the accumulation count.
        penalty, penalty_logs = controller.penalty()
        if penalty.requires_grad:
            scaler.scale(penalty).backward()

        mask_grad_norm = 0.0
        grad_norm = torch.tensor(0.0)
        if cfg.train.grad_clip > 0 or (mask_params and mask_clip > 0):
            scaler.unscale_(optimizer)
            if cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(weight_params, cfg.train.grad_clip)
            if mask_params and mask_clip > 0:
                mask_grad_norm = float(torch.nn.utils.clip_grad_norm_(mask_params, mask_clip))
        scaler.step(optimizer)
        scaler.update()

        ce_mean = ce_sum / accum
        running_ce += ce_mean
        running_n += 1
        step1 = step + 1

        if step1 % cfg.train.log_every_steps == 0 or step1 == 1:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_s = tokens_per_step * running_n / max(dt, 1e-6)
            metrics = {
                "train/ce": running_ce / running_n,
                "train/ppl": math.exp(min(20.0, running_ce / running_n)),
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
                "perf/tokens_per_s": tok_per_s,
                "perf/ms_per_step": 1000 * dt / running_n,
                "perf/tokens_seen": step1 * tokens_per_step,
            }
            metrics.update(penalty_logs)
            sp = controller.stats()
            metrics.update(sp)
            bn = bottleneck.stats()
            metrics.update(bn)
            if mask_params:
                metrics["sparsity/mask_grad_norm"] = mask_grad_norm
            metrics["train/loss"] = metrics["train/ce"] + sum(
                v for k, v in penalty_logs.items() if k.endswith("penalty")
            )

            line = (
                f"step {step1:>6}/{cfg.train.max_steps} | loss {metrics['train/loss']:.4f} "
                f"| ce {metrics['train/ce']:.4f} | ppl {metrics['train/ppl']:7.2f} "
                f"| lr {lr:.2e}"
            )
            if controller.enabled:
                line += (
                    f" | beta {sp['sparsity/beta']:.3g}"
                    f" | dens_soft {sp['sparsity/density_soft']:.4f}"
                    f" | dens_hard {sp['sparsity/density_hard']:.4f}"
                    f" | trans {sp['sparsity/transition_frac']:.4f}"
                )
                key = "sparsity/threshold_mean" if cfg.sparsity.method == "ltp" else "sparsity/s_mean"
                if key in sp:
                    line += f" | {key.split('/')[1]} {sp[key]:.3g}"
                if "sparsity/gate_mean_topk" in sp:
                    line += (
                        f" | gate {sp['sparsity/gate_mean_topk']:.3f}"
                        f" | turn {sp['sparsity/turnover']:.4f}"
                    )
            if "bottleneck/temperature" in bn:
                line += (
                    f" | t {bn['bottleneck/temperature']:.3g}"
                    f" | t/std {bn['bottleneck/temperature_rel']:.3g}"
                    f" | neff {bn['bottleneck/n_eff_realized']:.1f}"
                    f" | dK {bn['bottleneck/budget_residual']:.1e}"
                )
            elif bn:  # the hard baseline runs no solver
                line += f" | gap {bn['bottleneck/score_gap']:.3g}"
            line += (
                f" | gnorm {float(grad_norm):.2f}"
                f" | {human(tok_per_s)} tok/s | {metrics['perf/ms_per_step']:.0f} ms/step"
            )
            logger.log(step1, metrics, console=line)
            last_metrics = metrics
            running_ce, running_n = 0.0, 0

        if cfg.train.validate_every_steps and (
            step1 % cfg.train.validate_every_steps == 0 or step1 == cfg.train.max_steps
        ):
            val = evaluate(model, val_stream, micro_bs, cfg.train.val_batches, device, dtype)
            metrics = {"val/ce": val["ce"], "val/ppl": val["ppl"]}
            line = f"step {step1:>6} | val ce {val['ce']:.4f} | val ppl {val['ppl']:.2f}"
            if controller.enabled and cfg.sparsity.eval_hard_mask:
                with controller.hard_mask():
                    hard = evaluate(
                        model, val_stream, micro_bs, cfg.train.val_batches, device, dtype
                    )
                metrics["val_hard/ce"] = hard["ce"]
                metrics["val_hard/ppl"] = hard["ppl"]
                metrics["val_hard/density"] = controller.stats()["sparsity/density_hard"]
                line += (
                    f" | hard ce {hard['ce']:.4f} | hard ppl {hard['ppl']:.2f}"
                    f" | density {metrics['val_hard/density']:.4f}"
                )
            metrics.update({f"layer_{k}": v for k, v in controller.layer_densities().items()})
            if bottleneck.enabled and cfg.activation_bottleneck.log_diagnostics:
                metrics.update(log_feature_usage(logger, bottleneck, step1))
            logger.log(step1, metrics, console=line)
            best_val = min(best_val, val["ce"])
            last_metrics.update(metrics)

        if cfg.train.sample_every_steps and step1 % cfg.train.sample_every_steps == 0:
            texts = sample(model, cfg, device, dtype, step=step1)
            if texts:
                for i, text in enumerate(texts, 1):
                    print(f"[sample {i}/{len(texts)}] {text}")
                logger.log_text(
                    step1,
                    "samples",
                    "\n\n".join(f"**{i}.** {t}" for i, t in enumerate(texts, 1)),
                )

        if cfg.train.checkpoint_every_steps and (
            step1 % cfg.train.checkpoint_every_steps == 0 or step1 == cfg.train.max_steps
        ):
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            path = os.path.join(run_dir, f"ckpt_step{step1}.pt")
            save_checkpoint(path, cfg, base, optimizer, step1, extra={"metrics": last_metrics})
            save_checkpoint(
                os.path.join(run_dir, "latest.pt"),
                cfg,
                base,
                optimizer,
                step1,
                extra={"metrics": last_metrics},
            )
            prune_old_checkpoints(run_dir, cfg.train.keep_last_checkpoints)
            print(f"[ckpt] saved {path}")

    logger.close()
    summary = {"best_val_ce": best_val, **last_metrics}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def usage_figure(usage: Dict[str, "torch.Tensor"], k: int, n_features: int):
    """Rank-frequency curve of feature usage: the shape of the utilisation.

    Sorted descending and normalised by the uniform rate ``k/n``, on log-log
    axes, one line per bottlenecked layer.  A flat line at 1.0 would be perfectly
    even usage; the real curves are steeply Zipfian, and what matters is how far
    the tail falls -- features below ~1e-2 receive essentially no gradient and
    are on their way to dying.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - matplotlib is optional
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
    uniform = k / n_features
    ranks = np.arange(1, n_features + 1)
    cmap = plt.get_cmap("viridis")
    names = sorted(usage)
    for i, name in enumerate(names):
        u = np.sort(usage[name].float().cpu().numpy())[::-1] / uniform
        ax.loglog(ranks, np.maximum(u, 1e-6), lw=1.1,
                  color=cmap(i / max(1, len(names) - 1)),
                  label=name.split(".")[1] if "." in name else name)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axvline(k, color="tab:red", ls=":", lw=0.9)
    ax.set_xlabel(f"feature rank (of {n_features})")
    ax.set_ylabel("selection rate / uniform")
    ax.set_title(f"feature usage, sorted  (K={k}, dashed = even, dotted = rank K)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig


def log_feature_usage(logger, bottleneck, step: int) -> Dict[str, float]:
    """Usage distribution to TensorBoard: histogram, sorted-usage plot, quantiles."""
    usage = bottleneck.usage_vectors()
    if not usage:
        return {}
    cfg = bottleneck.cfg
    uniform = cfg.k / cfg.n_features
    metrics: Dict[str, float] = {}
    pooled = []
    for name, u in usage.items():
        logger.log_histogram(step, f"usage/{name}", u / uniform)
        pooled.append(u)
    stacked = torch.stack(pooled) / uniform
    for q in (0.5, 0.9, 0.99):
        metrics[f"bottleneck/usage_p{int(q * 100)}"] = float(
            torch.quantile(stacked.flatten().float(), q)
        )
    fig = usage_figure(usage, cfg.k, cfg.n_features)
    if fig is not None:
        logger.log_figure(step, "usage/sorted", fig)
    return metrics


def model_params(model, non_embedding: bool = False) -> int:
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    return base.num_parameters(non_embedding=non_embedding)


def sampling_generator(device: torch.device, seed: int) -> Optional[torch.Generator]:
    """A dedicated RNG for sampling, so the samples are comparable across runs.

    Without it, generation draws from the global RNG, whose state at step N
    depends on everything the run happened to consume beforehand -- with
    ``dropout: 0.0`` nothing in the training loop touches it, so samples do line
    up across runs, but that is an accident that any non-zero dropout breaks.
    """
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen
    except Exception:  # pragma: no cover - some backends have no device RNG
        return None


def sample(model, cfg: Config, device, dtype, step: int = 0) -> Optional[List[str]]:
    """``sample_count`` continuations of ``sample_prompt``, drawn as one batch."""
    try:
        from .tokenizer import build_tokenizer

        tok = build_tokenizer(cfg.data)
    except Exception as exc:  # pragma: no cover
        print(f"[sample] skipped ({exc})")
        return None
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    count = max(1, int(cfg.train.sample_count))
    prompt = torch.tensor(tok.encode(cfg.train.sample_prompt), dtype=torch.long, device=device)
    ids = prompt.unsqueeze(0).expand(count, -1).contiguous()
    with autocast_context(device, dtype):
        out = base.generate(
            ids,
            cfg.train.sample_tokens,
            temperature=0.8,
            top_k=50,
            generator=sampling_generator(device, cfg.train.seed + step),
        )
    base.train()
    return [tok.decode(row.tolist()).replace("\n", " ") for row in out]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a TinyStories LM with weight sparsity")
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    args, overrides = parser.parse_known_args(argv)
    cfg = load_config(args.config, overrides)
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
