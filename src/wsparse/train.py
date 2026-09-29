"""Training loop.

    python -m wsparse.train --config configs/ltp_base.yaml [--train.lr=6e-4 ...]
"""

from __future__ import annotations

import argparse
import builtins
import glob
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import Config, config_from_dict, load_config
from .data import build_streams, load_meta
from .model import build_model
from .optim import build_optimizer, count_parameter_groups, lr_at, set_lr
from .bottleneck import ActivationBottleneckController, apply_activation_bottleneck
from .utils import Logger, autocast_context, human, resolve_device, resolve_dtype, set_seed


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def ddp_env() -> Tuple[int, int, int]:
    """``(world_size, rank, local_rank)`` from torchrun's environment.

    ``(1, 0, 0)`` when launched as a plain process, so every caller can treat
    the single-GPU path as world size 1 rather than as a special case.
    """
    return (int(os.environ.get("WORLD_SIZE", "1")),
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("LOCAL_RANK", "0")))


def unwrap_model(model):
    """Peel ``torch.compile`` and DDP wrappers to the plain module."""
    m = model
    while True:
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif isinstance(m, torch.nn.parallel.DistributedDataParallel):
            m = m.module
        else:
            return m


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


def load_for_inference(path: str, device: str = "cpu"):
    """Rebuild a model (+ bottleneck) from a checkpoint.

    Returns ``(model, cfg, bottleneck_controller)`` -- the third element was
    the weight-sparsity controller before the 2026-09-28 cleanup; analysis
    callers that unpacked and ignored it are unaffected, and the probe tooling
    gets the bottleneck controller it actually wants.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(payload["config"])
    model = build_model(cfg.model)
    bottleneck = apply_activation_bottleneck(
        model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, cfg, bottleneck


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def log_bottleneck_geometry(counts: Dict[str, Any], cfg) -> None:
    """The bottleneck's init geometry, from what ``md_init_`` measured.

    Printed once at initialization.  ``md_init_`` returns the frame statistics
    for the first bottleneck only -- they are identical by construction across
    layers, and an SVD per layer is not free at 500M scale -- so that is what
    this reports.
    """
    if "bottleneck_init" not in counts:
        return  # no bottleneck in this model
    print(f"[train] bottleneck init: {counts['bottleneck_init']}"
          f"  d_model={cfg.model.d_model} d_bottleneck={counts['d_bottleneck']}"
          f"  K={counts['k']} J={counts['j']} K_eff={counts['k_eff']:g}"
          f"  decoder_scale={counts['decoder_scale_mode']} "
          f"g_D={counts['g_D']:.6g}")
    labels = {"encoder": "encoder rows", "decoder": "decoder cols",
              "decoder_effective": "decoder cols x g_D"}
    for side, label in labels.items():
        st = counts.get(side)
        if st is None:
            continue
        print(f"[train]   {label:22s} mean |.|^2 {st['mean_sq_norm']:.6f}"
              f"  ||W||_F {st['frobenius']:.4f}"
              f"  sv [{st['sv_min']:.4f}, {st['sv_max']:.4f}]"
              f"  gram rel err {st['gram_rel_err']:.2e}")


def train(cfg: Config, on_step: Optional[Callable[..., None]] = None,
          should_stop: Optional[Callable[[], Optional[str]]] = None) -> Dict[str, float]:
    """Train ``cfg``.  ``on_step(step, model, bottleneck, optimizer)``, when given,
    is called at the top of every optimizer step -- before the gradients for that
    step are accumulated -- so a caller can measure the model mid-training without
    holding checkpoints.  See ``interpretability/probe_early_training.py``.

    ``should_stop()``, when given, is polled once per optimizer step (after
    logging/eval/checkpointing) and a non-None reason ends the run cleanly:
    the loop breaks on EVERY rank in the same step -- the flag is all-reduced
    under DDP, which an exception could never be -- and the summary records
    the reason.  This is how scripts/train_guard.py stops diverged runs.
    """
    # ---- distributed context (torchrun) ---------------------------------- #
    # One process per GPU under `torchrun --nproc_per_node=N`; plain single-
    # process launches see world == 1 and none of the branches below fire.
    # Per-rank state is deliberate where it exists: the data stream RNG is
    # offset by rank (each rank sees different batches), gate buffers such as
    # usage_ema evolve per rank (broadcast_buffers=False), and everything
    # user-visible -- logging, validation, checkpoints, samples, the dumped
    # config -- is rank 0 only.
    world, rank, local_rank = ddp_env()
    is_main = rank == 0
    ddp_initialized_here = False
    if world > 1:
        # The configured device picks both the per-rank device and the
        # backend. Deciding from torch.cuda.is_available() instead would
        # promote an explicit `device: cpu` to cuda on a GPU box, and would
        # hand rank r a `cuda:r` that need not exist (fewer GPUs than ranks).
        wants_cuda = resolve_device(cfg.train.device).type == "cuda"
        if not dist.is_initialized():
            dist.init_process_group("nccl" if wants_cuda else "gloo")
            ddp_initialized_here = True
        if wants_cuda:
            torch.cuda.set_device(local_rank)
            cfg.train.device = f"cuda:{local_rank}"
    if not is_main:
        # local shadow: silences every print in this function on ranks > 0
        print = lambda *a, **k: None  # noqa: E731
    else:
        print = builtins.print

    set_seed(cfg.train.seed)  # identical across ranks: model init must agree
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = load_meta(cfg.data.data_dir)
    cfg.model.vocab_size = int(meta["vocab_size"])
    # rank-offset stream seed: each rank draws its own training batches; the
    # validation stream is read with deterministic offsets on rank 0 only, so
    # its RNG never matters.
    train_stream, val_stream = build_streams(
        cfg.data, seed=cfg.train.seed + 7919 * rank)

    model = build_model(cfg.model).to(device)
    bottleneck = apply_activation_bottleneck(
        model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps
    )
    model.to(device)  # bottleneck parameters created on cpu -> move again

    if cfg.model.decouple:
        # Magnitude-direction decoupling (arXiv:2606.25971): re-initialize the
        # finished model (bottlenecks included) onto the spheres the optimizer
        # will hold, then build the decoupled optimizer.  Overrides every other
        # init field, and no norm-constrained parameter gets weight decay.
        from .decouple import build_decoupled_optimizer, md_init_

        if dtype is torch.float16:
            raise ValueError(
                "decouple=True with float16 GradScaler is untested; use bfloat16")
        counts = md_init_(model, cfg.model.decouple_gains)
        print(f"[train] magnitude-direction decoupling: gains="
              f"{cfg.model.decouple_gains}, re-initialized {counts['matrix']} "
              f"matrices to their c_F spheres and {counts['embed']} embedding "
              f"tables to unit rows")
        log_bottleneck_geometry(counts, cfg)
        if cfg.train.weight_decay:
            # Not an error: the sphere replaces decay under this method, so a
            # configured value is ignored rather than rejected -- but say so,
            # because the dumped config.yaml will still show the configured
            # number, which is not what the run applied.
            print(f"[train] decouple=True IGNORES train.weight_decay="
                  f"{cfg.train.weight_decay}: no parameter receives weight "
                  f"decay (the norm constraint is the regularizer)")
        optimizer = build_decoupled_optimizer(
            model, cfg.train, gain_mode=cfg.model.decouple_gains,
        )
    else:
        if cfg.model.md_init:
            # The MD initialization without the MD optimizer: the same tensors a
            # decouple=True run starts from at the same seed (md_init_ is the
            # same call at the same point in the same construction sequence),
            # then ordinary AdamW below -- no gains, no re-projection, and
            # train.weight_decay applies as configured.
            from .decouple import md_init_

            counts = md_init_(model, cfg.model.decouple_gains)
            print(f"[train] md_init: re-initialized {counts['matrix']} matrices "
                  f"to their c_F norms and {counts['embed']} embedding tables to "
                  f"unit rows; training with plain AdamW "
                  f"(weight_decay={cfg.train.weight_decay})")
            log_bottleneck_geometry(counts, cfg)
        optimizer = build_optimizer(model, cfg.train)

    weight_params = [p for p in model.parameters()]

    run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    if is_main:
        logger = Logger(
            cfg.train.out_dir,
            cfg.train.run_name,
            config=cfg.to_dict(),
            wandb_project=cfg.train.wandb_project,
            wandb_entity=cfg.train.wandb_entity,
            tensorboard=cfg.train.tensorboard,
        )
        cfg.dump(os.path.join(run_dir, "config.yaml"))
    else:
        class _NullLogger:
            def log(self, *a, **k): pass
            def log_text(self, *a, **k): pass
            def log_figure(self, *a, **k): pass
            def close(self): pass
        logger = _NullLogger()
    if world > 1:
        dist.barrier()  # run_dir and config exist before anyone proceeds

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

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            # buffers stay per-rank: usage_ema, servo state and the forward
            # diagnostics are rank-local by design (initial construction
            # already synced parameters and buffers from rank 0)
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    if cfg.train.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype is torch.float16 and device.type == "cuda"))
    accum = cfg.train.grad_accum_steps
    micro_bs = int(cfg.train.micro_batch_size)
    # batch_size is PER RANK (as documented on the field); the global batch is
    # world x batch_size sequences, and throughput metrics report global tokens
    tokens_per_step = cfg.train.batch_size * cfg.data.seq_len * world

    print(
        f"[train] device={device} dtype={dtype} params={human(model_params(model))} "
        f"(non-emb {human(model_params(model, non_embedding=True))}) "
        f"batch={cfg.train.batch_size}x{cfg.data.seq_len} tok per rank "
        f"(micro {micro_bs} x accum {accum}"
        + (f" x world {world} = {human(tokens_per_step)} tok/step" if world > 1 else ")")
    )
    print(f"[train] param groups: {count_parameter_groups(optimizer)}")

    if bottleneck.enabled:
        cb = cfg.activation_bottleneck
        print(
            f"[train] activation bottleneck: {len(bottleneck.layers)} layers "
            f"({cfg.activation_bottleneck.layers}) {cb.placement} "
            f"N={cb.n_features} K={cb.k} J={cb.j} "
            f"({cb.selection_mode}, {cb.surrogate_mode}) "
            f"density={cb.k / cb.n_features:.3f} "
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

        if on_step is not None:
            # Deliberately before zero_grad: a probe that runs a backward of its
            # own leaves gradients in .grad, and the zero_grad below is then what
            # guarantees they can never reach the optimizer.  Called with the
            # schedules already set for `step`, and at step == start_step the
            # weights are still exactly at initialization.
            on_step(step, model, bottleneck, optimizer)

        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        for _ in range(accum):
            x, y = train_stream.batch(micro_bs, device)
            with autocast_context(device, dtype):
                _, ce = model(x, y)
            micro = ce
            ce_sum += ce.detach().float().item()
            scaler.scale(micro / accum).backward()

        grad_norm = torch.tensor(0.0)
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(weight_params, cfg.train.grad_clip)
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
            bn = bottleneck.stats()
            metrics.update(bn)
            metrics["train/loss"] = metrics["train/ce"]

            line = (
                f"step {step1:>6}/{cfg.train.max_steps} | loss {metrics['train/loss']:.4f} "
                f"| ce {metrics['train/ce']:.4f} | ppl {metrics['train/ppl']:7.2f} "
                f"| lr {lr:.2e}"
            )
            if "bottleneck/temperature" in bn:
                line += f" | t {bn['bottleneck/temperature']:.3g}"
                if "bottleneck/budget_residual" in bn:
                    line += f" | dK {bn['bottleneck/budget_residual']:.1e}"
            if "bottleneck/active_count" in bn:
                line += f" | L0 {bn['bottleneck/active_count']:.1f}"
                if "bottleneck/rb_cap_active_frac" in bn:
                    line += f" | cap {bn['bottleneck/rb_cap_active_frac']:.2f}"
            elif "bottleneck/score_gap" in bn:  # the hard baseline runs no solver
                line += f" | gap {bn['bottleneck/score_gap']:.3g}"
            elif bn:
                # k == n_features: every feature is active, so there is no
                # boundary between kept and dropped and no gap to report.
                line += f" | dense {bn.get('bottleneck/density', 1.0):.3g}"
            line += (
                f" | gnorm {float(grad_norm):.2f}"
                f" | {human(tok_per_s)} tok/s | {metrics['perf/ms_per_step']:.0f} ms/step"
            )
            logger.log(step1, metrics, console=line)
            last_metrics = metrics
            running_ce, running_n = 0.0, 0

        if cfg.train.validate_every_steps and (
            step1 % cfg.train.validate_every_steps == 0 or step1 == cfg.train.max_steps
        ) and is_main:
            val = evaluate(model, val_stream, micro_bs, cfg.train.val_batches, device, dtype)
            metrics = {"val/ce": val["ce"], "val/ppl": val["ppl"]}
            line = f"step {step1:>6} | val ce {val['ce']:.4f} | val ppl {val['ppl']:.2f}"
            if (bottleneck.enabled
                    and cfg.activation_bottleneck.surrogate_mode == "rblapsum_sf"):
                # val/ce above used the hard Top-K forward (hard_inference), so
                # it stays name-comparable across regimes; this second pass
                # evaluates the soft forward actually trained, z * p over the
                # Top(K+J) pool.
                gates = [mod.gate for _, mod in bottleneck.layers]
                prior = [g.hard_inference for g in gates]
                for g in gates:
                    g.hard_inference = False
                soft = evaluate(
                    model, val_stream, micro_bs, cfg.train.val_batches, device, dtype
                )
                for g, h in zip(gates, prior):
                    g.hard_inference = h
                metrics["val_soft/ce"] = soft["ce"]
                metrics["val_soft/ppl"] = soft["ppl"]
                line += f" | soft ce {soft['ce']:.4f}"
            if bottleneck.enabled and cfg.activation_bottleneck.log_diagnostics:
                metrics.update(log_feature_usage(logger, bottleneck, step1))
            logger.log(step1, metrics, console=line)
            best_val = min(best_val, val["ce"])
            last_metrics.update(metrics)

        if (cfg.train.sample_every_steps
                and step1 % cfg.train.sample_every_steps == 0 and is_main):
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
        ) and is_main:
            base = unwrap_model(model)
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

        # ---- coordinated stop (guards, --stop-step) ----------------------- #
        # Polled on every rank, all-reduced so every rank breaks in the same
        # step.  Under DDP an exception on one rank would hang the others in
        # the next gradient all-reduce; this flag is the supported way out.
        stop_reason = should_stop() if should_stop is not None else None
        if world > 1:
            flag = torch.tensor(
                [1 if stop_reason else 0],
                device=device if device.type == "cuda" else "cpu",
                dtype=torch.int32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            do_stop = bool(flag.item())
        else:
            do_stop = stop_reason is not None
        if do_stop:
            if stop_reason:
                print(f"[train] stop requested at step {step1}: {stop_reason}")
            last_metrics["stopped_at"] = step1
            if stop_reason:
                last_metrics["stopped_reason"] = stop_reason
            break

    stopped = "stopped_at" in last_metrics
    if cfg.train.final_val_batches and not stopped and is_main:
        # the full-holdout evaluation: many more batches than the routine
        # validation, run once at the end (skipped when a guard stopped the run)
        fv = evaluate(model, val_stream, micro_bs,
                      cfg.train.final_val_batches, device, dtype)
        logger.log(cfg.train.max_steps,
                   {"val_final/ce": fv["ce"], "val_final/ppl": fv["ppl"]},
                   console=(f"final validation ({cfg.train.final_val_batches} "
                            f"batches) | ce {fv['ce']:.4f} | ppl {fv['ppl']:.2f}"))
        last_metrics["val_final/ce"] = fv["ce"]
        last_metrics["val_final/ppl"] = fv["ppl"]

    logger.close()
    summary = {"best_val_ce": best_val, **last_metrics}
    if is_main:
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    if world > 1:
        dist.barrier()
        if ddp_initialized_here:
            dist.destroy_process_group()
    return summary


def usage_figure(usage: Dict[str, "torch.Tensor"], k: int, n_features: int):
    """Rank-frequency curve of feature usage: the shape of the utilization.

    Sorted descending and normalized by the uniform rate ``k/n``, on log-log
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
    return unwrap_model(model).num_parameters(non_embedding=non_embedding)


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
