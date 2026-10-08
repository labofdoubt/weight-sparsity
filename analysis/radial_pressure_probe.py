"""Radial pressure of the RBLapSum support term on the pool scores, per token.

At fixed checkpoints and identical training batches, in ``train()`` mode with
gradients enabled, the gate's backward is captured at every bottleneck as

    dL/dz      (gate input, the total: hard path + support term)      g_z
    dL/d~z     (gate output, before the mask)                         g_ztilde

so that, in value space, the support term alone is ``S = g_z - m * g_ztilde``
(the gate's backward is ``m * g + sign * g_s``) and the hard path over the
pool is ``m * g_ztilde``.  Both are mapped to *score* space (``s = |a|`` for
abs-TopK, so ``dL/ds = sign(a) dL/da``) and restricted to the Top-(K+J) pool
of the token.  With ``s`` the pool scores and ``sbar = s - mean(s)``:

    radial pressure       p   = <sbar, g_sur> / ||sbar||^2
    radial norm fraction  f   = <sbar, g_sur>^2 / (||sbar||^2 ||g_sur||^2)
    cosine                c   = cos(p * sbar, g_hard)
                              = sign(p) <sbar, g_hard> / (||sbar|| ||g_hard||)

``p`` is the coefficient of the support gradient along the centred pool
scores: a descent step rescales the pool's spread by ``(1 - lr * p)``, so a
negative ``p`` expands the spread (score-scale growth) and a positive one
contracts it.  ``f`` is the share of the support term's squared norm that is
radial.  ``c`` says whether that radial push agrees with the hard gradient's
own radial component.

Recorded per checkpoint, bottleneck and token, over a fixed set of training
batches (``TokenStream(train.bin).batch(..., deterministic_offset=o)``), plus
the boundary score ``b = s_(K+1)``, the pool span, the norms of the two
gradients, and the global parameter-gradient norm of the full backward and
of a hard-only backward (support scale 0) on the same batch -- the share of
the clipping budget the support term takes.

Under ``rblapsum_surrogate_scope=first_order*`` the backward is the two-pass
``first_order_backward`` the run trained with; the tensor hooks fire in both
passes and the total pass, the last one, is what is kept.

    python analysis/radial_pressure_probe.py --ckpt-dir /workspace/runs/<run> \
        --data-dir /workspace/data/tinystories --out /workspace/analysis/radial/<run>.npz

Output: one ``.npz`` with ``<step>/<layer>/<name>`` arrays over tokens
(``p``, ``f``, ``c``, ``b``, ``span``, ``gsur_norm``, ``ghard_norm``) and a
``.json`` of per-step, per-layer summaries and the gradient norms.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck.controller import _PLACEMENT_ATTR, parse_placements  # noqa: E402
from wsparse.bottleneck.rblapsum import first_order_backward  # noqa: E402
from wsparse.data import TokenStream  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402


def checkpoint_step(path: str) -> int:
    m = re.search(r"ckpt_step(\d+)\.pt$", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read a step number from {path!r}")
    return int(m.group(1))


def find_bottlenecks(model, cfg):
    placements = parse_placements(cfg.activation_bottleneck.placement)
    found = []
    for i, block in enumerate(model.blocks):
        for name in placements:
            mod = getattr(block, _PLACEMENT_ATTR[name], None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            found.append((f"blocks.{i}" if len(placements) == 1 else f"blocks.{i}.{name}", mod))
    if not found:
        raise ValueError("this checkpoint has no activation bottleneck installed")
    return found


def backward_with_capture(model, cfg, bottlenecks, idx, targets, grads: bool = True):
    """One train-mode forward+backward; returns (loss, {(name, li): tensor}, grad_norm)."""
    grabbed: dict = {}
    handles = []

    def make_pre(li):
        def pre(module, inputs):
            a = inputs[0]
            grabbed[("a", li)] = a.detach()
            if a.requires_grad:
                a.register_hook(lambda g, li=li: grabbed.__setitem__(("g_z", li), g.detach()))
        return pre

    def make_post(li):
        def post(module, inputs, output):
            if output.requires_grad:
                output.register_hook(
                    lambda g, li=li: grabbed.__setitem__(("g_ztilde", li), g.detach()))
        return post

    for li, (_, mod) in enumerate(bottlenecks):
        handles.append(mod.gate.register_forward_pre_hook(make_pre(li)))
        handles.append(mod.gate.register_forward_hook(make_post(li)))
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(idx, targets)
    scope = str(getattr(cfg.activation_bottleneck, "rblapsum_surrogate_scope", "pool"))
    if scope.startswith("first_order"):
        first_order_backward(loss, model.tok_emb.weight)
    else:
        loss.backward()
    for h in handles:
        h.remove()
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), grabbed, sq ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", help="run directory holding ckpt_step*.pt")
    ap.add_argument("--ckpt", action="append", default=[], help="explicit checkpoint(s)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--batch", type=int, default=8, help="sequences per fixed batch")
    ap.add_argument("--offsets", default="0,1", help="deterministic batch offsets, comma-separated")
    ap.add_argument("--out", required=True, help="output .npz path (a .json goes next to it)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = list(args.ckpt)
    if args.ckpt_dir:
        ckpts += glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt"))
    ckpts = sorted(set(ckpts), key=checkpoint_step)
    if not ckpts:
        raise SystemExit("no checkpoints given")
    offsets = [int(o) for o in args.offsets.split(",") if o.strip()]
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    arrays: dict = {}
    summary: dict = {"checkpoints": [], "offsets": offsets, "batch": args.batch,
                     "split": args.split, "steps": {}}
    batches = None
    for path in ckpts:
        step = checkpoint_step(path)
        model, cfg, _ = load_for_inference(path, device=str(device))
        bottlenecks = find_bottlenecks(model, cfg)
        ab = cfg.activation_bottleneck
        k, j = int(ab.k), int(ab.j)
        if ab.selection_mode != "abs_topk":
            raise SystemExit("the score-space mapping below assumes selection_mode=abs_topk")
        if batches is None:
            seq_len = int(cfg.data.seq_len)
            stream = TokenStream(os.path.join(args.data_dir, f"{args.split}.bin"), seq_len, seed=0)
            batches = [stream.batch(args.batch, device, deterministic_offset=o) for o in offsets]
            summary.update(run=os.path.basename(os.path.dirname(path)), k=k, j=j,
                           seq_len=seq_len, n_layers=len(bottlenecks),
                           scope=str(getattr(ab, "rblapsum_surrogate_scope", "pool")),
                           kernel_width=str(getattr(ab, "rblapsum_kernel_width", "fixed")),
                           temperature=float(ab.temperature),
                           support_strength=getattr(ab, "rblapsum_support_strength", None),
                           support_scale=float(ab.rblapsum_support_scale))
        per_layer = {li: {n: [] for n in ("p", "f", "c", "b", "span", "gsur_norm", "ghard_norm")}
                     for li in range(len(bottlenecks))}
        gnorm_total, gnorm_hard, losses = [], [], []
        for idx, targets in batches:
            loss, grabbed, gn = backward_with_capture(model, cfg, bottlenecks, idx, targets)
            losses.append(loss); gnorm_total.append(gn)
            # hard-only backward on the same batch: support scale 0 (strength off)
            saved = [(m.gate.rblapsum_support_scale, m.gate.rblapsum_support_strength)
                     for _, m in bottlenecks]
            for _, m in bottlenecks:
                m.gate.rblapsum_support_scale, m.gate.rblapsum_support_strength = 0.0, None
            _, grabbed_h, gn_h = backward_with_capture(model, cfg, bottlenecks, idx, targets)
            for (_, m), (sc, st) in zip(bottlenecks, saved):
                m.gate.rblapsum_support_scale, m.gate.rblapsum_support_strength = sc, st
            gnorm_hard.append(gn_h)

            for li in range(len(bottlenecks)):
                a = grabbed[("a", li)].double().reshape(-1, grabbed[("a", li)].shape[-1])
                gz = grabbed[("g_z", li)].double().reshape(a.shape)
                gzt = grabbed[("g_ztilde", li)].double().reshape(a.shape)
                s = a.abs()
                order = torch.argsort(s, dim=-1, descending=True)
                pool = order[:, : k + j]                                  # (tokens, K+J)
                sgn = torch.sign(a)
                active = torch.zeros_like(a).scatter_(1, order[:, :k], 1.0)
                S_val = gz - active * gzt                                 # support term, value space
                g_sur_full = sgn * S_val                                  # score space
                g_hard_full = sgn * active * gzt
                # the support term lives on the pool by construction
                off = g_sur_full.clone().scatter_(1, pool, 0.0)
                leak = float(off.abs().max()) / max(float(g_sur_full.abs().max()), 1e-30)
                # sanity: the hard-only pass must carry no support term
                gz_h = grabbed_h[("g_z", li)].double().reshape(a.shape)
                gzt_h = grabbed_h[("g_ztilde", li)].double().reshape(a.shape)
                resid = float((gz_h - active * gzt_h).abs().max()) / max(float(gz_h.abs().max()), 1e-30)
                if li == 0 and len(gnorm_total) == 1:
                    print(f"  step {step}: support leak off-pool {leak:.2e}, "
                          f"hard-only residual {resid:.2e}")
                sp = torch.gather(s, 1, pool)
                g_sur = torch.gather(g_sur_full, 1, pool)
                g_hard = torch.gather(g_hard_full, 1, pool)
                sbar = sp - sp.mean(dim=1, keepdim=True)
                sb2 = (sbar * sbar).sum(1)
                dot_s = (sbar * g_sur).sum(1)
                dot_h = (sbar * g_hard).sum(1)
                n_sur = g_sur.norm(dim=1)
                n_hard = g_hard.norm(dim=1)
                p = dot_s / sb2.clamp_min(1e-30)
                f = dot_s.pow(2) / (sb2 * n_sur.pow(2)).clamp_min(1e-60)
                c = torch.sign(p) * dot_h / (sb2.sqrt() * n_hard).clamp_min(1e-30)
                L = per_layer[li]
                L["p"].append(p.cpu().numpy()); L["f"].append(f.cpu().numpy())
                L["c"].append(c.cpu().numpy())
                L["b"].append(sp[:, k].cpu().numpy())
                L["span"].append((sp[:, k] - sp[:, k + j - 1]).cpu().numpy())
                L["gsur_norm"].append(n_sur.cpu().numpy()); L["ghard_norm"].append(n_hard.cpu().numpy())
        st_sum = {"loss": float(np.mean(losses)), "grad_norm_total": gnorm_total,
                  "grad_norm_hard_only": gnorm_hard, "layers": {}}
        for li, L in per_layer.items():
            for n, parts in L.items():
                arr = np.concatenate(parts)
                arrays[f"{step}/{li}/{n}"] = arr.astype(np.float32)
            q = lambda x: [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])]
            st_sum["layers"][str(li)] = {
                "p_mean": float(np.mean(np.concatenate(L["p"]))), "p_q": q(np.concatenate(L["p"])),
                "f_mean": float(np.mean(np.concatenate(L["f"]))), "f_q": q(np.concatenate(L["f"])),
                "c_mean": float(np.mean(np.concatenate(L["c"]))), "c_q": q(np.concatenate(L["c"])),
                "b_mean": float(np.mean(np.concatenate(L["b"]))),
                "span_mean": float(np.mean(np.concatenate(L["span"]))),
                "gsur_norm_mean": float(np.mean(np.concatenate(L["gsur_norm"]))),
                "ghard_norm_mean": float(np.mean(np.concatenate(L["ghard_norm"]))),
            }
        summary["steps"][str(step)] = st_sum
        summary["checkpoints"].append(path)
        print(f"step {step:>6}  loss {st_sum['loss']:.4f}  |grad| total {np.mean(gnorm_total):.3f} "
              f"hard-only {np.mean(gnorm_hard):.3f}  " + "  ".join(
                  f"L{li}: p {v['p_q'][2]:+.3g} f {v['f_q'][2]:.3f} c {v['c_q'][2]:+.2f} b {v['b_mean']:.3g}"
                  for li, v in st_sum["layers"].items() if li in ("1", "3", "5")))
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    np.savez_compressed(args.out, **arrays)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {args.out} and {os.path.splitext(args.out)[0]}.json")


if __name__ == "__main__":
    main()
