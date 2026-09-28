"""Per-layer activation/gradient diagnostics from a checkpoint.

Loads a saved checkpoint, runs ONE real forward+backward on a validation
batch, and records per bottlenecked block:

  - mean and median |z| over the K selected (top-K) features   ("topk_mean/med")
  - mean x^2 of the residual stream entering the bottleneck    ("stream_sq")
  - mean |dL/dz| over the bottleneck's raw pre-gate activations,
    over all N features and over the K active ones             ("gz_all/act")
  - mean |dL/dh| over the MLP hidden activations (post-GELU)   ("gmlp")

Usage:
  python analysis/layer_diag.py --ckpt <ckpt.pt> --data-dir <dir> \
      --out diag.json [--batch 8] [--offset 4242]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.config import config_from_dict  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset", type=int, default=4242)
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    cfg.data.data_dir = args.data_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model)
    bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                     max_steps=cfg.train.max_steps)
    model.load_state_dict(payload["model"])
    model.to(device).train()  # train mode: gradients as during training
    k = int(cfg.activation_bottleneck.k)
    print(f"[diag] {os.path.basename(args.ckpt)} step={payload['step']} "
          f"K={k} layers={len(bn.layers)}")

    cap = {}
    hooks = []
    for li, (lbl, mod) in enumerate(bn.layers):
        def pre(mod_, inp, li=li):
            cap[("x", li)] = inp[0].detach().float()
        hooks.append(mod.register_forward_pre_hook(pre))

        def gate_pre(mod_, inp, li=li):
            z = inp[0]
            cap[("z", li)] = z.detach().float()
            if z.requires_grad:
                z.register_hook(lambda g, li=li:
                                cap.__setitem__(("gz", li), g.detach().float()))
        hooks.append(mod.gate.register_forward_pre_hook(gate_pre))

    # MLP hidden activations: input of fc2 (post-nonlinearity)
    blocks = model.blocks if hasattr(model, "blocks") else model.transformer.blocks
    for li, blk in enumerate(blocks):
        def fc2_pre(mod_, inp, li=li):
            h = inp[0]
            cap[("h", li)] = h.detach().float()
            if h.requires_grad:
                h.register_hook(lambda g, li=li:
                                cap.__setitem__(("gh", li), g.detach().float()))
        hooks.append(blk.mlp.fc2.register_forward_pre_hook(fc2_pre))

    _, val_stream = build_streams(cfg.data, seed=cfg.train.seed)
    x, y = val_stream.batch(args.batch, device,
                            deterministic_offset=args.offset)
    _, loss = model(x, y)
    loss.backward()
    for h in hooks:
        h.remove()
    print(f"[diag] batch CE {float(loss):.4f}")

    n_layers = len(blocks)
    out = {"step": int(payload["step"]), "ce": float(loss), "k": k,
           "layers": []}
    for li in range(n_layers):
        rec = {"layer": li}
        if ("z", li) in cap:
            z = cap[("z", li)].reshape(-1, cap[("z", li)].shape[-1])
            topv = z.abs().topk(k, dim=-1).values
            rec["topk_mean"] = float(topv.mean())
            rec["topk_med"] = float(topv.median())
            x_in = cap[("x", li)]
            rec["stream_sq"] = float((x_in ** 2).mean())
            if ("gz", li) in cap:
                gz = cap[("gz", li)].reshape(-1, z.shape[-1])
                rec["gz_all"] = float(gz.abs().mean())
                act_idx = z.abs().topk(k, dim=-1).indices
                rec["gz_act"] = float(gz.gather(1, act_idx).abs().mean())
        if ("gh", li) in cap:
            rec["gmlp"] = float(cap[("gh", li)].abs().mean())
            rec["h_sq"] = float((cap[("h", li)] ** 2).mean())
        out["layers"].append(rec)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
