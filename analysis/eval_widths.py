"""Validation CE of one checkpoint at several hard support widths K'.

For a model trained with any gate, evaluate the hard Top-K' forward for a list
of K' (the gates' ``k`` is set to K' for the evaluation and restored).  Under
``stochastic_width`` training the model saw every K' in [K, K+J]; this
measures what it does at each of them at inference.

  python analysis/eval_widths.py --ckpt latest.pt --widths 32 64 128 256 --out x.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surrogate_probe import build  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.train import evaluate  # noqa: E402
from wsparse.utils import resolve_dtype  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--widths", type=int, nargs="+", required=True)
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--micro", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tree = copy.deepcopy(payload["config"])
    cfg, model, bn = build(tree, dev, payload)
    cfg.data.data_dir = args.data_dir
    _, val = build_streams(cfg.data, seed=cfg.train.seed)
    dtype = resolve_dtype(cfg.train.dtype, dev)
    gates = [mod.gate for _, mod in bn.layers]
    k0 = gates[0].k
    out = {"source": args.ckpt, "step": int(payload["step"]), "k_train": k0,
           "j": int(cfg.activation_bottleneck.j), "widths": {}}
    for kp in args.widths:
        for g in gates:
            g.k = int(kp)
            g.m = g.k + g.j
        model.eval()
        r = evaluate(model, val, args.micro, args.batches, dev, dtype)
        out["widths"][str(kp)] = r["ce"]
        print(f"[eval_widths] K'={kp}: val ce {r['ce']:.4f}")
    for g in gates:
        g.k = k0
        g.m = g.k + g.j
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
