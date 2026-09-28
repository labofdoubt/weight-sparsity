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
import copy
import json
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.config import apply_overrides, config_from_dict  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.model import build_model  # noqa: E402


def token_sq(t: torch.Tensor) -> torch.Tensor:
    """Per-token squared L2 norm over the feature axis, flattened to [B*T]."""
    f = t.detach().float()
    return (f * f).sum(-1).reshape(-1)


def gain(sq_in: torch.Tensor, sq_out: torch.Tensor, forward: bool) -> dict:
    """Gain along the direction of propagation, from two [B*T] vectors.

    ``sq_in`` / ``sq_out`` are always keyed by NETWORK POSITION -- the module's
    input side and output side -- whichever direction is being measured, so a
    reader never has to work out what "in" meant for a backward pass.  The gain
    then runs with the flow: out/in going forward, in/out going backward.
    """
    eps = torch.finfo(torch.float32).tiny
    num, den = (sq_out, sq_in) if forward else (sq_in, sq_out)
    return {
        "gain": float(num.mean() / den.mean().clamp_min(eps)),
        "gain_median_token": float((num / den.clamp_min(eps)).median()),
        "sq_input_side": float(sq_in.mean()),
        "sq_output_side": float(sq_out.mean()),
    }


def cb_k(cfg) -> int:
    return int(cfg.activation_bottleneck.k)


def overridden(tree: dict, overrides) -> dict:
    """The run config with CLI-style overrides applied (never in place)."""
    if not overrides:
        return tree
    tree = copy.deepcopy(tree)
    apply_overrides(tree, [f"--{o.lstrip('-')}" for o in overrides])
    return tree


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
    ap.add_argument("--pnorm-gamma", default="1.0",
                    help="scale for the post-RMSNorm's gains; \"auto\" runs a "
                         "second pass with gamma = 1/sqrt(that layer's measured "
                         "backward gain), which puts the norm's gradient gain "
                         "at 1 (it enters both directions as gamma^2)")
    ap.add_argument("--md-alpha", default="1.0",
                    help="total scale for each bottleneck's OUTPUT, put into "
                         "the MD gains (md_spread_gain_) and split equally "
                         "across the four gain vectors of its two projections; "
                         "\"auto\" resolves to sqrt(d_model / K), the factor "
                         "that makes the bottleneck's gradient gain 1")
    ap.add_argument("--proj-scale", type=float, default=1.0,
                    help="multiply BOTH bottleneck projections' weights by this "
                         "after initialization, so the product of their stds "
                         "scales by its square; the identity is 1.0")
    ap.add_argument("--override", action="append", default=[],
                    help="config override applied to the loaded run config, "
                         "e.g. --override activation_bottleneck.k=64 "
                         "(repeatable; same syntax and coercion as the CLI)")
    args = ap.parse_args()

    assert bool(args.ckpt) != bool(args.init_config), \
        "give exactly one of --ckpt / --init-config"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = config_from_dict(overridden(payload["config"], args.override))
        cfg.data.data_dir = args.data_dir
        model = build_model(cfg.model)
        bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                         max_steps=cfg.train.max_steps)
        model.load_state_dict(payload["model"])
        step = int(payload["step"])
    else:
        from wsparse.utils import set_seed
        cfg = config_from_dict(overridden(json.load(open(args.init_config)),
                                          args.override))
        cfg.data.data_dir = args.data_dir
        set_seed(cfg.train.seed)
        model = build_model(cfg.model)
        bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                         max_steps=cfg.train.max_steps)
        if cfg.model.decouple:
            from wsparse.decouple import md_init_
            md_init_(model, cfg.model.decouple_gains)
        step = 0
    md_alpha = (math.sqrt(cfg.model.d_model / cb_k(cfg))
                if str(args.md_alpha) == "auto" else float(args.md_alpha))
    if md_alpha != 1.0:
        if not cfg.model.decouple:
            raise SystemExit("--md-alpha needs model.decouple=true (the gains "
                             "it spreads into are MD's)")
        from wsparse.decouple import md_spread_gain_
        per_matrix = math.sqrt(md_alpha)  # two matrices in series
        gains = []
        for _, mod in bn.layers:
            gains.append(md_spread_gain_(mod.in_proj.weight, per_matrix,
                                         cfg.model.decouple_gains))
            if not getattr(mod, "tied", False):
                md_spread_gain_(mod.out_proj.weight, per_matrix,
                                cfg.model.decouple_gains)
        print(f"[gain] md gain spread: alpha={md_alpha:.4f} on each bottleneck's "
              f"output = {per_matrix:.4f} per matrix = {gains[0]:.4f} per gain "
              f"vector ({cfg.model.decouple_gains})")
    if args.proj_scale != 1.0:
        # A tied decoder IS the encoder transposed, so scaling in_proj already
        # scales both sides; touching out_proj as well would square the factor.
        with torch.no_grad():
            for _, mod in bn.layers:
                mod.in_proj.weight.mul_(args.proj_scale)
                if not getattr(mod, "tied", False):
                    mod.out_proj.weight.mul_(args.proj_scale)
        print(f"[gain] scaled both projections by {args.proj_scale:.4f} "
              f"(std product x{args.proj_scale ** 2:.4f})")
    model.to(device).train()

    cb = cfg.activation_bottleneck
    print(f"[gain] {'init' if not args.ckpt else os.path.basename(args.ckpt)} "
          f"step={step} L={cfg.model.n_layers} d={cfg.model.d_model} "
          f"N={cb.n_features} K={cb.k} {cb.placement} "
          f"post_norm={cb.post_norm} bottlenecks={len(bn.layers)}")

    _, val_stream = build_streams(cfg.data, seed=cfg.train.seed)
    xb, yb = val_stream.batch(args.batch, device, deterministic_offset=args.offset)

    def measure():
        """One forward+backward; returns (ce, fwd, bwd) of per-token sq norms.

        ``fwd[(point, li)]`` and ``bwd[(point, li)]`` are [B*T] vectors at the
        three probe points x (module input), y (decoder output = the norm's
        input) and z (module output).
        """
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

        model.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        for h in hooks:
            h.remove()
        return float(loss.detach()), fwd, bwd

    ce, fwd, bwd = measure()
    print(f"[gain] batch CE {ce:.4f}  tokens {args.batch * cfg.data.seq_len}")

    # ---- post-norm gamma ------------------------------------------------- #
    # "auto" is a second pass: the RMSNorm's gain enters both directions as
    # gamma^2, so gamma = 1/sqrt(measured backward gain at gamma=1) puts that
    # gain at 1 -- per layer, since it drifts slightly with depth.
    gammas = {}
    if str(args.pnorm_gamma) != "1.0":
        eps = torch.finfo(torch.float32).tiny
        with torch.no_grad():
            for li, (_, mod) in enumerate(bn.layers):
                if isinstance(mod.post_norm, nn.Identity):
                    continue
                if str(args.pnorm_gamma) == "auto":
                    g_in = bwd[("y", li)].mean()          # module-input side
                    g_out = bwd[("z", li)].mean()         # module-output side
                    gamma = float((g_out / g_in.clamp_min(eps)).sqrt())
                else:
                    gamma = float(args.pnorm_gamma)
                mod.post_norm.weight.mul_(gamma)
                gammas[li] = gamma
        vals = sorted(gammas.values())
        print(f"[gain] post-norm gamma: {len(gammas)} layers, "
              f"{vals[0]:.4f} .. {vals[-1]:.4f} "
              f"(geo-mean {math.exp(sum(map(math.log, vals)) / len(vals)):.4f})")
        ce, fwd, bwd = measure()
        print(f"[gain] batch CE {ce:.4f} after the gamma rescale")

    out = {
        "step": step, "ce": ce, "k": int(cb.k), "j": int(cb.j),
        "n_features": int(cb.n_features), "d_model": int(cfg.model.d_model),
        "n_layers": int(cfg.model.n_layers), "placement": str(cb.placement),
        "post_norm": bool(cb.post_norm), "surrogate_mode": str(cb.surrogate_mode),
        "batch": int(args.batch), "seq_len": int(cfg.data.seq_len),
        "offset": int(args.offset), "proj_scale": float(args.proj_scale),
        "md_alpha": float(md_alpha),
        "pnorm_gamma": str(args.pnorm_gamma),
        "pnorm_gamma_per_layer": {str(k): v for k, v in sorted(gammas.items())},
        "layers": [],
    }
    for li in range(len(bn.layers)):
        rec = {"layer": li}
        # bottleneck: x -> y  (encoder through decoder, post-norm excluded)
        if ("y", li) in fwd:
            rec["fwd_bottleneck"] = gain(fwd[("x", li)], fwd[("y", li)], True)
        if ("y", li) in bwd and ("x", li) in bwd:
            rec["bwd_bottleneck"] = gain(bwd[("x", li)], bwd[("y", li)], False)
        # post-norm: y -> z
        if ("z", li) in fwd and ("y", li) in fwd:
            rec["fwd_post_norm"] = gain(fwd[("y", li)], fwd[("z", li)], True)
        if ("z", li) in bwd and ("y", li) in bwd:
            rec["bwd_post_norm"] = gain(bwd[("y", li)], bwd[("z", li)], False)
        out["layers"].append(rec)

    def compounded(field):
        vals = [r[field]["gain"] for r in out["layers"] if field in r]
        if not vals:
            return None
        logs = [math.log(v) for v in vals if v > 0]
        return {"n": len(vals), "product": math.exp(sum(logs)),
                "geometric_mean": math.exp(sum(logs) / len(logs)),
                "min": min(vals), "max": max(vals)}

    out["summary"] = {f: compounded(f) for f in
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
