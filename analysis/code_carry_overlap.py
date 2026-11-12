"""How much of the carried code survives each bottleneck under code_residual.

Under ``code_residual`` the stream a block hands on is the decoded code,
``x_{l+1} = D c_{l+1}`` with ``c_{l+1} = (c_l + E Delta_l) * m_{l+1}``, so a
block's input reaches later blocks only through the carried code ``c_l``, on
the coordinates the next support ``m_{l+1}`` keeps.  For a config (at
initialization, or from a checkpoint) this runs one deterministic forward on
real validation windows and reports, per consecutive pair of bottlenecks,

    overlap = |S_l intersect S_{l+1}| / K
    carried = ||c_l * m_{l+1}||^2 / ||c_l||^2      (energy of c_l that survives)

averaged over tokens.

    python analysis/code_carry_overlap.py --config CFG --val-bin val.bin [--set k=v ...]
    python analysis/code_carry_overlap.py --ckpt run/ckpt_step1000.pt --val-bin val.bin
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
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.model import build_model  # noqa: E402


def build(cfg, ckpt=None):
    torch.manual_seed(cfg.train.seed)
    model = build_model(cfg.model)
    ctl = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                      max_steps=cfg.train.max_steps)
    if ckpt is None and cfg.model.decouple:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), ctl


@torch.no_grad()
def measure(model, ctl, idx):
    codes = []
    hooks = [layer.gate.register_forward_hook(lambda m, a, out: codes.append(out.float()))
             for _, layer in ctl.layers]
    model(idx)
    for h in hooks:
        h.remove()
    k = ctl.layers[0][1].gate.k
    rows = []
    for a, b in zip(codes[:-1], codes[1:]):
        sa, sb = a != 0, b != 0
        overlap = (sa & sb).sum(-1).float().mean().item() / k
        carried = ((a * sb).pow(2).sum(-1) / a.pow(2).sum(-1).clamp_min(1e-30)).mean().item()
        rows.append((overlap, carried))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="", help="append {label, k, overlap, carried} to this file")
    args = ap.parse_args()
    ckpt = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = config_from_dict(ckpt["config"])
    else:
        cfg = load_config(args.config, args.set)
    model, ctl = build(cfg, ckpt)
    tok = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(tok) - args.seq - 1, size=args.batch)
    idx = torch.from_numpy(np.stack([tok[s:s + args.seq].astype(np.int64) for s in starts]))
    rows = measure(model, ctl, idx)
    ab = cfg.activation_bottleneck
    print(f"{args.label or ab.selection_mode} K={ab.k}  overlap per pair "
          + " ".join(f"{o:.3f}" for o, _ in rows)
          + f" | mean {np.mean([o for o, _ in rows]):.3f};  carried energy "
          + " ".join(f"{c:.3f}" for _, c in rows)
          + f" | mean {np.mean([c for _, c in rows]):.3f}")
    if args.json:
        data = json.load(open(args.json)) if os.path.exists(args.json) else []
        data.append({"label": args.label, "k": ab.k, "set": args.set,
                     "overlap": [o for o, _ in rows], "carried": [c for _, c in rows]})
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)


if __name__ == "__main__":
    main()
