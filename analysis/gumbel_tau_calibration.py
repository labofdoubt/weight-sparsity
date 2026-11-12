"""Gumbel temperature calibration at initialization (gumbel_policy).

For a config at its initialization (the same construction as ``train``:
build_model, the bottleneck, md_init_ under decouple), runs one deterministic
forward on real validation windows, captures every gate's selection scores
(the learned score s under learned_score_topk, |a| under abs_topk) and the
decoded stream each bottleneck hands on, and reports per temperature tau the
exchange fraction of a Gumbel-Top-K draw,

    xi(tau) = 1 - |S_tau  intersect  TopK(s)| / K,     S_tau = TopK_K(s / tau + G),

averaged over tokens and gates (pool: all N, or the Top(K+J) candidates), plus
the RMS of the decoded streams.  The scores of later gates come from the
deterministic forward, so xi describes each gate's own draw given clean inputs.

    python analysis/gumbel_tau_calibration.py --config CFG --val-bin val.bin \\
        --set activation_bottleneck.k=32 --taus 0.01 0.03 0.1 0.3 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.bottleneck.gumbel_policy import gumbel_topk  # noqa: E402
from wsparse.config import load_config  # noqa: E402
from wsparse.model import build_model  # noqa: E402


def build(cfg):
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg.model)
    ctl = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                      max_steps=cfg.train.max_steps)
    if cfg.model.decouple:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    return model.eval(), ctl


@torch.no_grad()
def capture(model, ctl, idx):
    scores, streams = [], []
    hooks = []
    for _, layer in ctl.layers:
        gate = layer.gate

        def pre(mod, args, _g=gate):
            s = args[0]
            scores.append(s.abs() if _g.selection_mode == "abs_topk" else s)
        hooks.append(gate.register_forward_pre_hook(pre))

        def post(mod, args, out):
            streams.append(out.float().pow(2).mean().sqrt().item())
        hooks.append(layer.out_proj.register_forward_hook(post))
    model(idx)
    for h in hooks:
        h.remove()
    return scores, streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--taus", nargs="+", type=float, required=True)
    ap.add_argument("--pool", choices=["all", "top_kj"], default="all")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ab = cfg.activation_bottleneck
    model, ctl = build(cfg)
    tok = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(tok) - args.seq - 1, size=args.batch)
    idx = torch.from_numpy(np.stack([tok[s:s + args.seq].astype(np.int64) for s in starts]))
    scores, streams = capture(model, ctl, idx)
    k, m = ab.k, ab.k + ab.j
    pool = m if args.pool == "top_kj" else None
    gen = torch.Generator().manual_seed(0)
    out = {"config": args.config, "set": args.set, "selection_mode": ab.selection_mode,
           "k": k, "j": ab.j, "pool": args.pool, "stream_rms": streams,
           "score_std": [float(s.float().std(-1).mean()) for s in scores],
           "score_gap_k": [float((torch.topk(s.float(), k + 1, -1).values[..., k - 1]
                                  - torch.topk(s.float(), k + 1, -1).values[..., k]).mean())
                           for s in scores],
           "xi": {}}
    for tau in args.taus:
        per_gate = []
        for s in scores:
            clean = torch.zeros_like(s, dtype=torch.bool).scatter(
                -1, torch.topk(s.float(), k, -1).indices, True)
            xs = []
            for _ in range(args.draws):
                d = gumbel_topk(s.float(), k, tau, gen, pool=pool)
                got = torch.zeros_like(clean).scatter(-1, d.idx, True)
                xs.append(1.0 - (got & clean).sum(-1).float().mean().item() / k)
            per_gate.append(float(np.mean(xs)))
        out["xi"][str(tau)] = {"mean": float(np.mean(per_gate)), "per_gate": per_gate}
    print(f"{ab.selection_mode} K={k} J={ab.j} pool={args.pool}  "
          f"stream rms per gate: {' '.join(f'{r:.3g}' for r in streams)}")
    print(f"  score std per gate: {' '.join(f'{r:.3g}' for r in out['score_std'])}")
    for tau, d in out["xi"].items():
        print(f"  tau {float(tau):<8g} xi mean {d['mean']:.3f}   per gate "
              + " ".join(f"{x:.2f}" for x in d["per_gate"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
