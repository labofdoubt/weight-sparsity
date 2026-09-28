"""Per-layer forward and backward gain of the bottleneck and of its post-norm.

Every bottlenecked block carries three probe points on the stream,

    x --[ in_proj -> gate -> out_proj ]--> y --[ post_norm ]--> z

so "the bottleneck" here is the whole encode/select/decode sandwich, measured
between its own input and the decoder's output, and "the post-norm" is the
RMSNorm that follows it inside the same module.  One real forward+backward on a
validation batch (train mode, so the surrogate path is live) supplies both the
activations and the gradients at those three points.

Reported per layer, as a gain along the direction of propagation:

    forward    G_fwd  = <||out||^2> / <||in||^2>
    backward   G_bwd  = <||dL/din||^2> / <||dL/dout||^2>

where <.> averages the per-token squared L2 norm over the feature axis across
all B*T tokens -- an energy ratio, not a ratio of per-token ratios.  The median
of the per-token ratios is reported alongside, since a few tokens can carry a
large share of the norm.  G > 1 means the module amplifies in that direction.

Note that `residual_out` gives the LAST block's bottleneck no post-norm of its
own (its output already feeds `norm_f`), so that layer's post-norm entries are
null rather than 1.0.

Usage:
  python analysis/module_gain.py --init-config <run config.json> \
      --data-dir <dir> --out gain.json [--batch 8] [--offset 4242]
  python analysis/module_gain.py --ckpt <ckpt.pt> --data-dir <dir> --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.config import config_from_dict  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.model import build_model  # noqa: E402


def token_sq(t: torch.Tensor) -> torch.Tensor:
    """Per-token squared L2 norm over the feature axis, flattened to [B*T]."""
    f = t.detach().float()
    return (f * f).sum(-1).reshape(-1)


def ratio(num: torch.Tensor, den: torch.Tensor) -> dict:
    """Energy gain and the median per-token gain, from two [B*T] vectors."""
    eps = torch.finfo(torch.float32).tiny
    per_token = num / den.clamp_min(eps)
    return {
        "gain": float(num.mean() / den.mean().clamp_min(eps)),
        "gain_median_token": float(per_token.median()),
        "in_sq_mean": float(den.mean()),
        "out_sq_mean": float(num.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init-config", default="",
                    help="measure at initialization instead: build from this "
                         "run config with train()'s construction sequence "
                         "(seed, build, bottleneck, md_init), no weights loaded")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset", type=int, default=4242)
    args = ap.parse_args()

    assert bool(args.ckpt) != bool(args.init_config), \
        "give exactly one of --ckpt / --init-config"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = config_from_dict(payload["config"])
        cfg.data.data_dir = args.data_dir
        model = build_model(cfg.model)
        bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                         max_steps=cfg.train.max_steps)
        model.load_state_dict(payload["model"])
        step = int(payload["step"])
    else:
        from wsparse.utils import set_seed
        cfg = config_from_dict(json.load(open(args.init_config)))
        cfg.data.data_dir = args.data_dir
        set_seed(cfg.train.seed)
        model = build_model(cfg.model)
        bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                         max_steps=cfg.train.max_steps)
        if cfg.model.decouple:
            from wsparse.decouple import md_init_
            md_init_(model, cfg.model.decouple_gains)
        step = 0
    model.to(device).train()

    cb = cfg.activation_bottleneck
    print(f"[gain] {'init' if not args.ckpt else os.path.basename(args.ckpt)} "
          f"step={step} L={cfg.model.n_layers} d={cfg.model.d_model} "
          f"N={cb.n_features} K={cb.k} {cb.placement} "
          f"post_norm={cb.post_norm} bottlenecks={len(bn.layers)}")

    # fwd[(point, li)] and bwd[(point, li)] hold [B*T] per-token squared norms
    fwd, bwd, hooks = {}, {}, []

    def keep(store, point, li):
        def fn(t):
            store[(point, li)] = token_sq(t)
        return fn

    for li, (_, mod) in enumerate(bn.layers):
        has_norm = not isinstance(mod.post_norm, nn.Identity)

        def mod_pre(mod_, inp, li=li):
            x = inp[0]
            keep(fwd, "x", li)(x)
            if x.requires_grad:
                x.register_hook(keep(bwd, "x", li))
        hooks.append(mod.register_forward_pre_hook(mod_pre))

        def mod_post(mod_, inp, out, li=li, has_norm=has_norm):
            keep(fwd, "z" if has_norm else "y", li)(out)
            if out.requires_grad:
                out.register_hook(keep(bwd, "z" if has_norm else "y", li))
        hooks.append(mod.register_forward_hook(mod_post))

        if has_norm:  # y = the decoder's output = the norm's input
            def norm_pre(mod_, inp, li=li):
                y = inp[0]
                keep(fwd, "y", li)(y)
                if y.requires_grad:
                    y.register_hook(keep(bwd, "y", li))
            hooks.append(mod.post_norm.register_forward_pre_hook(norm_pre))

    _, val_stream = build_streams(cfg.data, seed=cfg.train.seed)
    xb, yb = val_stream.batch(args.batch, device, deterministic_offset=args.offset)
    _, loss = model(xb, yb)
    loss.backward()
    for h in hooks:
        h.remove()
    print(f"[gain] batch CE {float(loss):.4f}  tokens {args.batch * cfg.data.seq_len}")

    out = {
        "step": step, "ce": float(loss), "k": int(cb.k), "j": int(cb.j),
        "n_features": int(cb.n_features), "d_model": int(cfg.model.d_model),
        "n_layers": int(cfg.model.n_layers), "placement": str(cb.placement),
        "post_norm": bool(cb.post_norm), "surrogate_mode": str(cb.surrogate_mode),
        "batch": int(args.batch), "seq_len": int(cfg.data.seq_len),
        "offset": int(args.offset), "layers": [],
    }
    for li in range(len(bn.layers)):
        rec = {"layer": li}
        # bottleneck: x -> y  (encoder through decoder, post-norm excluded)
        if ("y", li) in fwd:
            rec["fwd_bottleneck"] = ratio(fwd[("y", li)], fwd[("x", li)])
        if ("y", li) in bwd and ("x", li) in bwd:
            rec["bwd_bottleneck"] = ratio(bwd[("x", li)], bwd[("y", li)])
        # post-norm: y -> z
        if ("z", li) in fwd and ("y", li) in fwd:
            rec["fwd_post_norm"] = ratio(fwd[("z", li)], fwd[("y", li)])
        if ("z", li) in bwd and ("y", li) in bwd:
            rec["bwd_post_norm"] = ratio(bwd[("y", li)], bwd[("z", li)])
        out["layers"].append(rec)

    def compound(field):
        vals = [r[field]["gain"] for r in out["layers"] if field in r]
        if not vals:
            return None
        logs = [math.log(v) for v in vals if v > 0]
        return {"n": len(vals), "product": math.exp(sum(logs)),
                "geometric_mean": math.exp(sum(logs) / len(logs)),
                "min": min(vals), "max": max(vals)}

    out["summary"] = {f: compound(f) for f in
                      ("fwd_bottleneck", "fwd_post_norm",
                       "bwd_bottleneck", "bwd_post_norm")}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for f_, s in out["summary"].items():
        if s:
            print(f"[gain] {f_:15s} geo-mean {s['geometric_mean']:.4g}  "
                  f"product over {s['n']} layers {s['product']:.4g}  "
                  f"(min {s['min']:.4g}, max {s['max']:.4g})")
    print(f"[gain] wrote {args.out}")


if __name__ == "__main__":
    main()
