"""Does rope cost anything over the learned position table?  Measure, don't guess.

Times forward and forward+backward for the same architecture under both
``pos_encoding`` modes, at the training shape (micro-batch x seq_len) and under
the training autocast dtype, with CUDA events and a warmup. The two models are
identically configured apart from the flag, and the bottleneck can be included
or not -- included by default, since that is what this project actually trains.

    python scripts/bench_pos_encoding.py \
        --config /workspace/ckpt/dc_rout_soft_k32_j64/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.utils import autocast_context, resolve_device, resolve_dtype  # noqa: E402


def build(cfg, pos_encoding: str, device):
    import copy

    c = copy.deepcopy(cfg)
    c.model.pos_encoding = pos_encoding
    torch.manual_seed(0)
    model = build_model(c.model).to(device)
    if c.activation_bottleneck.enabled:
        apply_activation_bottleneck(model, c.activation_bottleneck,
                                    max_steps=c.train.max_steps)
        model.to(device)
    return model


def bench(model, x, y, device, dtype, backward: bool, iters: int, warmup: int):
    model.train(backward)
    if not backward:
        model.eval()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(warmup + iters):
        if backward:
            model.zero_grad(set_to_none=True)
        if i == warmup:
            torch.cuda.synchronize()
        start.record()
        if backward:
            with autocast_context(device, dtype):
                _, loss = model(x, y)
            loss.backward()
        else:
            with torch.no_grad(), autocast_context(device, dtype):
                model(x, y)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return float(t.median()), float(t.quantile(0.9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-bottleneck", action="store_true")
    a = ap.parse_args()

    cfg = (config_from_dict(json.load(open(a.config)))
           if a.config.endswith(".json") else load_config(a.config))
    if a.no_bottleneck:
        cfg.activation_bottleneck.enabled = False
    device = resolve_device(cfg.train.device)
    dtype = resolve_dtype(cfg.train.dtype, device)
    B, T = int(cfg.train.micro_batch_size), int(cfg.data.seq_len)
    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
    print(f"shape {B}x{T}, dtype {dtype}, device {device}, "
          f"bottleneck={'on' if cfg.activation_bottleneck.enabled else 'off'}, "
          f"{a.iters} iters after {a.warmup} warmup")
    print(f"{'mode':>9} {'params':>13} | {'fwd med ms':>10} {'p90':>8} "
          f"| {'fwd+bwd med ms':>14} {'p90':>8}")
    results = {}
    for mode in ("learned", "rope"):
        model = build(cfg, mode, device)
        n = sum(p.numel() for p in model.parameters())
        f_med, f_p90 = bench(model, x, y, device, dtype, False, a.iters, a.warmup)
        b_med, b_p90 = bench(model, x, y, device, dtype, True, a.iters, a.warmup)
        results[mode] = (f_med, b_med)
        print(f"{mode:>9} {n:>13,} | {f_med:10.3f} {f_p90:8.3f} "
              f"| {b_med:14.3f} {b_p90:8.3f}")
        del model
        torch.cuda.empty_cache()
    fl, bl = results["learned"]
    fr, br = results["rope"]
    print(f"\nrope / learned:  forward {fr/fl:.4f}x   forward+backward {br/bl:.4f}x")
    print(f"(above 1.0 means rope is slower by that factor)")


if __name__ == "__main__":
    main()
