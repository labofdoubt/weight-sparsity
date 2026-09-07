"""Measure the surrogate's scale-direction gradient along a live trajectory.

The LapSum VJP is exact at fixed (b, t); under temperature_scale_mode="relative"
t is a function of the scores and the soft mask is exactly invariant to a common
rescaling of a row's scores, so the true gradient has no component along the
score direction.  This probe re-runs training from a run's own config (same
seed, nothing saved) and, on a fixed held-out batch every N steps, measures the
component the implementation keeps anyway:

    S_J(row) = sum over the J band of (dL/da_i * a_i)

A gradient step moves the row's score scale by -lr * S_J, so S_J < 0 means
descent inflates that row's scale.  Each measurement is repeated with every
gate flipped to temperature_scale_mode="absolute" -- same weights, same batch --
to isolate the relative-t contribution.  See
docs/relative-temperature-divergence.md for the investigation this decided.

    python analysis/diagnose_relative_temperature.py \
        --config /workspace/runs/<run>/config.json \
        --data-dir /workspace/data/tinystories \
        --out /workspace/analysis/diag_phantom.json --steps 1000 --every 50

Non-perturbation follows probe_early_training: the hook runs before the step's
zero_grad, and gate buffers and RNG are restored after each measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.train import train  # noqa: E402
from wsparse.bottleneck.controller import _PLACEMENT_ATTR  # noqa: E402


class StopProbing(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True, help="JSON file for the measurements")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layers to report (default: all bottlenecked)")
    a = ap.parse_args()

    cfg = (config_from_dict(json.load(open(a.config)))
           if a.config.endswith(".json") else load_config(a.config))
    cfg.data.data_dir = a.data_dir
    cfg.train.run_name = "diag_phantom"
    cfg.train.out_dir = "/tmp/diag_phantom"
    cfg.train.resume = ""
    cfg.train.tensorboard = False
    cfg.train.sample_every_steps = 10 ** 9
    cfg.train.validate_every_steps = 10 ** 9
    cfg.train.checkpoint_every_steps = 10 ** 9

    K = int(cfg.activation_bottleneck.k)
    M = K + int(cfg.activation_bottleneck.j)
    placement = _PLACEMENT_ATTR[cfg.activation_bottleneck.placement]

    tok = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    T = int(cfg.data.seq_len)
    x_np = np.stack([tok[i * T:(i + 1) * T] for i in range(a.batch)]).astype(np.int64)
    y_np = np.stack([tok[i * T + 1:(i + 1) * T + 1] for i in range(a.batch)]).astype(np.int64)
    rows_out: list = []

    def measure(model, device):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        mods = [(li, getattr(blk, placement)) for li, blk in enumerate(model.blocks)
                if getattr(blk, placement, None) is not None]
        cap: dict = {}
        handles = []

        def mk(li):
            def pre(m, inputs):
                t_in = inputs[0]
                cap[("a", li)] = t_in.detach().float()
                if t_in.requires_grad:
                    t_in.register_hook(
                        lambda g, li=li: cap.__setitem__(("g", li), g.detach().float()))
            return pre

        for li, mod in mods:
            handles.append(mod.gate.register_forward_pre_hook(mk(li)))
        _, loss = model(x, y)
        loss.backward()
        for h in handles:
            h.remove()
        out = {}
        for li, _ in mods:
            av = cap[("a", li)].reshape(-1, cap[("a", li)].shape[-1])
            gv = cap[("g", li)].reshape(-1, av.shape[-1])
            r = av.abs()
            order = torch.argsort(r, dim=-1, descending=True)
            S_J = (gv * av).gather(1, order[:, K:M]).sum(-1)
            R2 = (r.gather(1, order[:, :M]) ** 2).sum(-1)
            rate = -S_J / R2.clamp_min(1e-30)
            out[li] = dict(frac_inflate=float((S_J < 0).float().mean()),
                           rate_med=float(rate.median()),
                           rate_p90=float(rate.quantile(0.9)),
                           scale_p99=float(r.quantile(0.99)))
        return float(loss.detach()), out

    def on_step(step, model, bottleneck, optimizer):
        if step % a.every != 0 and step != 0:
            if step >= a.steps:
                raise StopProbing()
            return
        device = next(model.parameters()).device
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        gates = [getattr(blk, placement).gate for blk in model.blocks
                 if getattr(blk, placement, None) is not None]
        snaps = [(g.usage_ema.clone(), g.usage_steps.clone(), dict(g._forward_diag),
                  dict(g._usage_diag), dict(g._grad_sink)) for g in gates]
        was_training = model.training
        model.train()

        ce, rel = measure(model, device)
        model.zero_grad(set_to_none=True)
        modes = [g.temperature_scale_mode for g in gates]
        for g in gates:
            g.temperature_scale_mode = "absolute"
        _, ab = measure(model, device)
        for g, m in zip(gates, modes):
            g.temperature_scale_mode = m
        model.zero_grad(set_to_none=True)

        for g, s in zip(gates, snaps):
            g.usage_ema.copy_(s[0]); g.usage_steps.copy_(s[1])
            g._forward_diag, g._usage_diag, g._grad_sink = s[2], s[3], s[4]
        model.train(was_training)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)

        layers = a.layers if a.layers else sorted(rel)
        for li in layers:
            rows_out.append(dict(step=step, layer=li, ce=ce,
                                 **{f"rel_{k}": v for k, v in rel[li].items()},
                                 **{f"abs_{k}": v for k, v in ab[li].items()}))
        top = max(rel)
        print(f"[phantom] step {step:>5}  L{top} scale_p99 {rel[top]['scale_p99']:9.3g}  "
              f"rel: frac_inflate {rel[top]['frac_inflate']:.2f} "
              f"rate_med {rel[top]['rate_med']:+.2e}  "
              f"abs-counterfactual: frac {ab[top]['frac_inflate']:.2f} "
              f"rate {ab[top]['rate_med']:+.2e}", flush=True)
        if step >= a.steps:
            raise StopProbing()

    try:
        train(cfg, on_step=on_step)
    except StopProbing:
        pass
    json.dump(rows_out, open(a.out, "w"))
    print(f"wrote {a.out} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
