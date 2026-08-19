"""Per-feature selection distribution of an activation bottleneck, from a checkpoint.

    python scripts/feature_usage.py --ckpt runs/bn_lapsum/latest.pt --batches 100

Counts are exact (accumulated from the same TopK the forward pass uses), not the
running EMA the training diagnostics report -- the EMA is a non-persistent
buffer and is not stored in checkpoints.  Writes the raw per-layer counts to
``<ckpt_dir>/feature_usage.npz`` so the distribution can be plotted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wsparse.bottleneck import SparseTopKBottleneck  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402
from wsparse.utils import autocast_context, resolve_device, resolve_dtype  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_for_inference(args.ckpt, device=str(device))
    dtype = resolve_dtype(cfg.train.dtype, device)
    model.eval()

    gates = [(n, m.gate) for n, m in model.named_modules() if isinstance(m, SparseTopKBottleneck)]
    if not gates:
        sys.exit("this checkpoint has no activation bottleneck")

    counts = {name: torch.zeros(g.n_features, dtype=torch.float64, device=device)
              for name, g in gates}
    handles = []
    for name, gate in gates:
        def hook(mod, inp, _out, _name=name):
            a = inp[0]
            scores = a.abs() if mod.selection_mode == "abs_topk" else a
            idx = torch.topk(scores, mod.k, dim=-1).indices.reshape(-1)
            counts[_name] += torch.bincount(idx, minlength=mod.n_features).double()
        handles.append(gate.register_forward_hook(hook))

    stream = TokenStream(
        os.path.join(cfg.data.data_dir, f"{args.split}.bin"), seq_len=cfg.data.seq_len, seed=0
    )
    stride = args.batch_size * (cfg.data.seq_len + 1)
    with torch.no_grad():
        for i in range(args.batches):
            x, _ = stream.batch(args.batch_size, device, deterministic_offset=i * stride)
            with autocast_context(device, dtype):
                model(x)
    for h in handles:
        h.remove()

    tokens = args.batches * args.batch_size * cfg.data.seq_len
    k, n = gates[0][1].k, gates[0][1].n_features
    uniform = k / n
    print(f"{os.path.basename(os.path.dirname(args.ckpt))}: {tokens:,} tokens, K={k} N={n}")
    print(f"{'layer':<12}{'dead%':>8}{'entropy':>9}{'max/unif':>10}{'p50/unif':>10}{'p99/unif':>10}")
    summary, raw = {}, {}
    for name, _ in gates:
        c = (counts[name] / tokens).cpu().numpy()  # selection rate per feature
        raw[name] = counts[name].cpu().numpy()
        p = c / max(c.sum(), 1e-30)
        ent = float(np.exp(-(p * np.log(np.clip(p, 1e-30, None))).sum()) / n)
        row = dict(
            dead=float((c < 0.01 * uniform).mean()),
            entropy=ent,
            max=float(c.max() / uniform),
            p50=float(np.median(c) / uniform),
            p99=float(np.quantile(c, 0.99) / uniform),
        )
        summary[name] = row
        print(f"{name:<12}{row['dead']*100:>7.1f}%{row['entropy']:>9.3f}"
              f"{row['max']:>10.2f}{row['p50']:>10.2f}{row['p99']:>10.2f}")

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "feature_usage.npz")
    np.savez_compressed(out, tokens=tokens, k=k, n_features=n, **raw)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"tokens": tokens, "k": k, "n_features": n, "layers": summary}, f, indent=2)
    print(f"\nraw counts -> {out}")


if __name__ == "__main__":
    main()
