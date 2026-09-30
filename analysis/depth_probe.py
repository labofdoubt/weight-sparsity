"""Per-layer signal propagation through a stack of stream bottlenecks.

One real forward+backward on a validation batch (train mode, grad enabled), at
initialization (``--init-config``, train()'s construction sequence at the run's
seed) or from a checkpoint (``--ckpt``).  Recorded per block ``l``:

  stream_rms        RMS of the stream entering block l (the input of norm1)
  stream_grad_rms   RMS of dL/dx_l at that point (the whole gradient reaching it)
  tok_cos           mean cosine similarity between the streams of different
                    positions of a sequence (1 = every position identical)
  g_<param>         RMS of the gradient of qkv / proj / fc1 / fc2 of block l
  sel_energy        mean_kept z^2 / mean_all z^2 at block l's bottleneck gate
                    (s^2; 4.02 for Gaussian codes at K/N = 1/8)
  fwd_gain          ||y||^2 / ||h||^2 of the bottleneck (decoder output over its
                    input, before any post-norm), token-averaged energies
  bwd_gain          ||dL/dh||^2 / ||dL/dy||^2 over the same span
  dW_rel_<param>    ||W - W_0|| / ||W_0||, with ``--init-reference`` (W_0 is the
                    run's own step-0 weight, rebuilt from its config and seed)

Works for code_residual models too: the stream is the decoded code the block
reads, and the bottleneck span is the gate's (the carry has no decoder).

  python analysis/depth_probe.py --init-config <config.json|yaml> --out x.json \
      [--set a.b=c ...]
  python analysis/depth_probe.py --ckpt <ckpt.pt> --out x.json [--init-reference]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.config import apply_overrides, config_from_dict  # noqa: E402
from wsparse.data import build_streams  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.utils import set_seed  # noqa: E402

PARAMS = {"qkv": "attn.qkv", "proj": "attn.proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}


def load_tree(path: str) -> dict:
    if path.endswith(".json"):
        return json.load(open(path))
    import yaml
    return yaml.safe_load(open(path))


def build_init(tree: dict, device: torch.device):
    """train()'s construction sequence: seed, build, bottleneck, move, md_init.

    md_init_ runs AFTER the move in train(), so on a GPU run it draws from the
    CUDA generator; re-initializing on the CPU gives statistically identical
    but different weights (||W - W_0|| / ||W_0|| then reads ~sqrt(2)).
    """
    cfg = config_from_dict(tree)
    set_seed(cfg.train.seed)
    model = build_model(cfg.model).to(device)
    bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                     max_steps=cfg.train.max_steps)
    model.to(device)
    if cfg.model.decouple or cfg.model.md_init:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    return cfg, model, bn


def tok_sq(t: torch.Tensor) -> torch.Tensor:
    f = t.detach().float()
    return (f * f).sum(-1).reshape(-1)


def rms(t) -> float:
    return float(t.detach().float().pow(2).mean().sqrt()) if t is not None else float("nan")


def token_cosine(x: torch.Tensor, pairs: int = 4096, gen=None) -> float:
    """Mean cosine between the stream vectors of two different positions of the
    same sequence (positions >= 16, so the first tokens' special role is out)."""
    B, T, _ = x.shape
    lo = min(16, T - 2)
    i = torch.randint(lo, T, (pairs,), generator=gen)
    j = torch.randint(lo, T, (pairs,), generator=gen)
    b = torch.randint(0, B, (pairs,), generator=gen)
    keep = i != j
    xf = x.detach().float()
    u, v = xf[b[keep], i[keep]], xf[b[keep], j[keep]]
    return float(torch.nn.functional.cosine_similarity(u, v, dim=-1).mean())


def common_energy(t: torch.Tensor, lo: int = 16):
    """(energy of the per-sequence mean over positions >= lo, mean energy per position).

    For streams x_t = c + u_t with the u_t independent, the first over the second
    is the common-mode fraction f = |c|^2 / (|c|^2 + |u|^2), which is also the
    expected cosine between two positions (up to a 1/T bias)."""
    tf = t.detach().float()[:, lo:]
    common = (tf.mean(1) ** 2).sum(-1).mean()
    total = (tf ** 2).sum(-1).mean()
    return float(common), float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init-config", default="")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    ap.add_argument("--init-reference", action="store_true",
                    help="also rebuild the run's step-0 weights and report ||W-W0||/||W0||")
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset", type=int, default=4242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert bool(args.ckpt) != bool(args.init_config), "give --ckpt or --init-config"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tree = copy.deepcopy(payload["config"])
        apply_overrides(tree, args.set)
        cfg = config_from_dict(tree)
        model = build_model(cfg.model)
        # the controller restores g_D from the config (it is not state)
        bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                         max_steps=cfg.train.max_steps)
        model.load_state_dict(payload["model"])
        step = int(payload["step"])
        del payload
    else:
        tree = load_tree(args.init_config)
        apply_overrides(tree, args.set)
        cfg, model, bn = build_init(tree, dev)
        step = 0
    cfg.data.data_dir = args.data_dir
    model.to(dev).train()

    ref = None
    if args.init_reference and args.ckpt:
        _, m0, _ = build_init(copy.deepcopy(tree), dev)
        ref = {n: p.detach().clone() for n, p in m0.named_parameters()}
        del m0

    blocks = list(model.blocks)
    n_layers = len(blocks)
    cap, hooks = {}, []
    # the stream entering block l = the input of norm1 (both forward paths)
    for li, blk in enumerate(blocks):
        def pre(mod_, inp, li=li):
            t = inp[0]
            if t.requires_grad:
                t.retain_grad()
            cap[("x", li)] = t
        hooks.append(blk.norm1.register_forward_pre_hook(pre))

    # the block's contribution Delta = attention + MLP output (residual_out
    # placement: no bottleneck inside the branches), both forward paths
    for li, blk in enumerate(blocks):
        hooks.append(blk.attn.register_forward_hook(
            lambda m_, i, o, li=li: cap.__setitem__(("a", li), o.detach())))
        hooks.append(blk.mlp.register_forward_hook(
            lambda m_, i, o, li=li: cap.__setitem__(("m", li), o.detach())))

    code_res = getattr(model, "code_entry", None) is not None
    mods = {}
    for label, mod in bn.layers:
        if label.startswith("blocks."):
            mods[int(label.split(".")[1])] = mod
    for li, mod in mods.items():
        def gate_pre(mod_, inp, li=li):
            z = inp[0]
            cap[("z", li)] = z.detach()
            if not code_res and z.requires_grad:
                z.retain_grad()
        hooks.append(mod.gate.register_forward_pre_hook(gate_pre))
        if not code_res:
            def mod_pre(mod_, inp, li=li):
                h = inp[0]
                if h.requires_grad:
                    h.retain_grad()
                cap[("h", li)] = h
            hooks.append(mod.register_forward_pre_hook(mod_pre))
            # the decoder output: post_norm's input, or the module's output
            if isinstance(mod.post_norm, torch.nn.Identity):
                def mod_post(mod_, inp, out, li=li):
                    if out.requires_grad:
                        out.retain_grad()
                    cap[("y", li)] = out
                hooks.append(mod.register_forward_hook(mod_post))
            else:
                def pn_pre(mod_, inp, li=li):
                    y = inp[0]
                    if y.requires_grad:
                        y.retain_grad()
                    cap[("y", li)] = y
                hooks.append(mod.post_norm.register_forward_pre_hook(pn_pre))

    _, val = build_streams(cfg.data, seed=cfg.train.seed)
    x, y = val.batch(args.batch, dev, deterministic_offset=args.offset)
    from wsparse.utils import autocast_context, resolve_dtype
    dtype = resolve_dtype(cfg.train.dtype, dev)
    with autocast_context(dev, dtype):
        _, loss = model(x, y)
    loss.backward()
    for h in hooks:
        h.remove()

    gen = torch.Generator().manual_seed(0)
    k = int(cfg.activation_bottleneck.k)
    out = {"step": step, "ce": float(loss), "k": k,
           "n_features": int(cfg.activation_bottleneck.n_features),
           "d_model": int(cfg.model.d_model), "n_layers": n_layers,
           "code_residual": code_res,
           "post_norm": bool(cfg.activation_bottleneck.post_norm),
           "value_shift": getattr(cfg.activation_bottleneck, "value_shift", "none"),
           "source": args.ckpt or args.init_config, "overrides": args.set,
           "layers": []}
    names = dict(model.named_parameters())
    for li, blk in enumerate(blocks):
        rec = {"layer": li}
        xs = cap.get(("x", li))
        if xs is not None:
            rec["stream_rms"] = rms(xs)
            rec["stream_grad_rms"] = rms(xs.grad)
            rec["tok_cos"] = token_cosine(xs, gen=gen)
            cx, ex = common_energy(xs)
            rec["f_stream"] = cx / ex
            if ("a", li) in cap and ("m", li) in cap:
                delta = cap[("a", li)].float() + cap[("m", li)].float()
                cd, ed = common_energy(delta)
                ca, _ = common_energy(cap[("a", li)])
                rec["r_delta"] = ed / ex          # ||Delta||^2 / ||x||^2
                rec["r_common"] = cd / ex         # its token-independent part
                rec["r_common_attn"] = ca / ex    # ... from attention alone
                ch, eh = common_energy(xs.detach().float() + delta)
                rec["f_h"] = ch / eh
        for short, attr in PARAMS.items():
            p = blk.get_submodule(attr).weight
            rec[f"g_{short}"] = rms(p.grad)
            if ref is not None:
                full = f"blocks.{li}.{attr}.weight"
                w0 = ref[full].to(p.device)
                rec[f"dW_rel_{short}"] = float((p.detach().float() - w0.float()).norm()
                                               / w0.float().norm())
        if li in mods:
            mod = mods[li]
            z = cap[("z", li)].float().reshape(-1, cap[("z", li)].shape[-1])
            kept = z.abs().topk(k, dim=-1).values
            rec["sel_energy"] = float((kept ** 2).mean() / (z ** 2).mean())
            if not code_res and ("h", li) in cap and ("y", li) in cap:
                h_, y_ = cap[("h", li)], cap[("y", li)]
                rec["fwd_gain"] = float(tok_sq(y_).mean() / tok_sq(h_).mean())
                if h_.grad is not None and y_.grad is not None:
                    rec["bwd_gain"] = float(tok_sq(h_.grad).mean()
                                            / tok_sq(y_.grad).mean())
            if ref is not None and not bool(cfg.activation_bottleneck.share_projections):
                for short, attr in (("enc", "in_proj"), ("dec", "out_proj")):
                    p = getattr(mod, attr).weight
                    full = [n for n, q in names.items() if q is p][0]
                    w0 = ref[full].to(p.device)
                    rec[f"dW_rel_{short}"] = float(
                        (p.detach().float() - w0.float()).norm() / w0.float().norm())
        out["layers"].append(rec)

    g = [r["g_fc1"] for r in out["layers"]]
    out["fc1_grad_last_over_first"] = g[-1] / g[0]
    out["per_layer_factor"] = math.exp((math.log(g[-1]) - math.log(g[0])) / (n_layers - 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    first, last = out["layers"][0], out["layers"][-1]
    print(f"[depth_probe] step {step} CE {out['ce']:.3f} | fc1 grad first {g[0]:.2e} "
          f"last {g[-1]:.2e} (x{out['per_layer_factor']:.3f}/layer) | tok_cos first "
          f"{first.get('tok_cos', float('nan')):.3f} last {last.get('tok_cos', float('nan')):.3f}"
          f" | sel_energy mean {sum(r.get('sel_energy', 0) for r in out['layers']) / n_layers:.2f}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
