"""Per-gate backward decomposition of a code-residual stack (hard vs surrogate).

For a ``code_residual`` model (``c_{l+1} = gate_l(c_l + alpha E Delta_l)``) at
initialization or from a checkpoint, one forward and one backward on a
validation batch (train mode, grad enabled), recording for every gate l (the
entry gate first, then the blocks):

  code_grad_rms    RMS of dL/dc_{l+1} at the gate's output (the carried code)
  in_grad_rms      RMS of dL/du_l at the gate's input u_l = c_l + alpha E Delta_l
  hard_grad_rms    RMS of the hard part  dL/dy * mask  of dL/du_l
  surr_grad_rms    RMS of the surrogate part  (dL/du_l - hard part)
  sigma            mean over tokens of ||surrogate part||^2 / ||hard part||^2
  boundary         mean rank boundary b = s_(K+1) of the gate
  kept_rms         RMS of the kept scores |u_i|, i in TopK
  n_window         mean number of pool members with |s_i - b| < T
  pool_grad_rms    RMS of dL/dy over the Top(K+J) pool (the upstream the
                   surrogate multiplies)
  b_over_t         mean b / T_eff (T_eff = T, or T * b under
                   rblapsum_relative_temperature): |u| kappa at the boundary is
                   half of it
  n_eff            mean kernel population (sum kappa)^2 / sum kappa^2 over the pool
  pi_surr          mean and median over tokens of the exact surrogate gain
                   ||L_kappa D_z||_F^2 of docs/scale-dynamics-note-neutral.tex
                   (pi_surr, pi_surr_med)
  supp_common_frac share of the support term's energy (score space, over the
                   pool) in its per-feature token mean -- the part that moves
                   one feature's score for every token at once
  push_usage_corr  correlation over features between that token-mean push
                   (descent direction, + = score up for all tokens) and the
                   feature's active frequency in the batch
  n_half           features active for more than half of the batch's tokens

Energies are per token over the N code coordinates, then averaged over tokens.
The gradient at the gate's input and output is captured with tensor hooks, so
nothing about the model's forward or backward changes.

  python analysis/surrogate_probe.py --init-config <yaml|json> --out x.json [--set a.b=c ...]
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
from wsparse.utils import autocast_context, resolve_dtype, set_seed  # noqa: E402


def load_tree(path):
    if path.endswith(".json"):
        return json.load(open(path))
    import yaml
    return yaml.safe_load(open(path))


def build(tree, device, ckpt=None):
    cfg = config_from_dict(tree)
    set_seed(cfg.train.seed)
    model = build_model(cfg.model).to(device)
    bn = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                     max_steps=cfg.train.max_steps)
    model.to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    elif cfg.model.decouple or cfg.model.md_init:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    return cfg, model, bn


def tok_energy(t):
    f = t.detach().float()
    return (f * f).sum(-1).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--data-dir", default="/workspace/data/tinystories")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--offset", type=int, default=4242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = None
    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tree = copy.deepcopy(payload["config"])
    else:
        tree = load_tree(args.init_config)
    apply_overrides(tree, args.set)
    cfg, model, bn = build(tree, dev, payload)
    cfg.data.data_dir = args.data_dir
    model.train()
    assert getattr(model, "code_entry", None) is not None, "needs a code_residual model"
    ab = cfg.activation_bottleneck
    k, t = int(ab.k), float(ab.temperature)
    rel_t = bool(getattr(ab, "rblapsum_relative_temperature", False))

    gates = [model.code_entry.gate] + [b.residual_out_bottleneck.gate for b in model.blocks]
    cap, hooks = {}, []
    for gi, gate in enumerate(gates):
        def pre(mod_, inp, gi=gi):
            u = inp[0]
            cap[("u", gi)] = u
            if u.requires_grad:
                u.register_hook(lambda g, gi=gi: cap.__setitem__(("du", gi), g.detach()))
        def post(mod_, inp, out, gi=gi):
            cap[("y", gi)] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda g, gi=gi: cap.__setitem__(("dy", gi), g.detach()))
        hooks.append(gate.register_forward_pre_hook(pre))
        hooks.append(gate.register_forward_hook(post))

    _, val = build_streams(cfg.data, seed=cfg.train.seed)
    x, y = val.batch(args.batch, dev, deterministic_offset=args.offset)
    dtype = resolve_dtype(cfg.train.dtype, dev)
    with autocast_context(dev, dtype):
        _, loss = model(x, y)
    if ab.rblapsum_surrogate_scope.startswith("first_order"):
        from wsparse.bottleneck.rblapsum import first_order_backward
        first_order_backward(loss, model.tok_emb.weight)
    else:
        loss.backward()
    for h in hooks:
        h.remove()

    out = {"ce": float(loss), "k": k, "j": int(ab.j), "temperature": t,
           "surrogate_mode": ab.surrogate_mode, "overrides": args.set,
           "source": args.ckpt or args.init_config, "gates": []}
    for gi in range(len(gates)):
        u = cap[("u", gi)].detach().float()
        du = cap.get(("du", gi))
        dy = cap.get(("dy", gi))
        rec = {"gate": gi}
        s = u.abs()
        top = s.topk(k + max(int(ab.j), 1), dim=-1)
        b = top.values[..., k]                        # (K+1)-st score
        kept = top.values[..., :k]
        mask = torch.zeros_like(u).scatter(-1, top.indices[..., :k], 1.0)
        rec["boundary"] = float(b.mean())
        rec["kept_rms"] = float(kept.pow(2).mean().sqrt())
        rec["u_rms"] = float(u.pow(2).mean().sqrt())
        pool_s = top.values
        t_eff = (t * b).clamp_min(1e-6) if rel_t else torch.full_like(b, t)
        rec["b_over_t"] = float((b / t_eff).mean())
        rec["n_window"] = float(((pool_s - b.unsqueeze(-1)).abs()
                                 < t_eff.unsqueeze(-1)).float().sum(-1).mean())
        # kernel population and the exact surrogate gain ||L_kappa D_z||_F^2
        kap = (torch.exp(-(pool_s - b.unsqueeze(-1)).abs() / t_eff.unsqueeze(-1))
               / (2 * t_eff.unsqueeze(-1)))
        z_sum, k_sq = kap.sum(-1, keepdim=True), kap.pow(2).sum(-1, keepdim=True)
        rec["n_eff"] = float((z_sum.pow(2) / k_sq).mean())
        zc = u.gather(-1, top.indices)
        pi = (zc.pow(2) * kap.pow(2) * ((1 - kap / z_sum).pow(2)
                                        + (k_sq - kap.pow(2)) / z_sum.pow(2))).sum(-1)
        rec["pi_surr"] = float(pi.mean())
        rec["pi_surr_med"] = float(pi.flatten().median())
        if du is not None and dy is not None:
            du, dy = du.float(), dy.float()
            hard = dy * mask
            surr = du - hard
            e_h, e_s = tok_energy(hard), tok_energy(surr)
            rec["code_grad_rms"] = float(dy.pow(2).mean().sqrt())
            rec["in_grad_rms"] = float(du.pow(2).mean().sqrt())
            rec["hard_grad_rms"] = float(hard.pow(2).mean().sqrt())
            rec["surr_grad_rms"] = float(surr.pow(2).mean().sqrt())
            rec["sigma"] = float((e_s / e_h.clamp_min(1e-30)).mean())
            pool_idx = top.indices
            rec["pool_grad_rms"] = float(dy.gather(-1, pool_idx).pow(2).mean().sqrt())
            # the support term's per-feature token mean (score space, pool only)
            q = pool_idx.shape[-1]
            n_feat = u.shape[-1]
            gs = (surr * u.sign()).gather(-1, pool_idx).reshape(-1, q).double()
            flat = pool_idx.reshape(-1)
            tot = torch.zeros(n_feat, dtype=gs.dtype, device=gs.device).index_add_(
                0, flat, gs.reshape(-1))
            cnt = torch.zeros(n_feat, dtype=gs.dtype, device=gs.device).index_add_(
                0, flat, torch.ones_like(gs).reshape(-1))
            mean = tot / cnt.clamp_min(1)
            e_all = float(gs.pow(2).sum())
            rec["supp_common_frac"] = (float((cnt * mean.pow(2)).sum()) / e_all
                                       if e_all > 0 else 0.0)
            usage = mask.reshape(-1, n_feat).mean(0).double()
            push = -mean
            sel = cnt > 0
            if int(sel.sum()) > 2 and float(push[sel].std()) > 0:
                pu = torch.stack([push[sel], usage[sel]])
                rec["push_usage_corr"] = float(torch.corrcoef(pu)[0, 1])
            rec["n_half"] = int((usage > 0.5).sum())
            rec["offsupport_grad_rms"] = float((dy * (1 - mask)).pow(2).mean().sqrt())
        out["gates"].append(rec)

    G = out["gates"]
    cg = [g.get("code_grad_rms") for g in G[1:] if g.get("code_grad_rms")]
    if len(cg) > 1:
        out["per_block_factor"] = math.exp((math.log(cg[-1]) - math.log(cg[0])) / (len(cg) - 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    sig = [g["sigma"] for g in G if "sigma" in g]
    print(f"[surrogate_probe] CE {out['ce']:.3f} per-block code-grad factor "
          f"x{out.get('per_block_factor', float('nan')):.3f}  sigma mean {sum(sig)/max(1,len(sig)):.3f}  "
          f"b mean {sum(g['boundary'] for g in G)/len(G):.2f}  kept_rms "
          f"{sum(g['kept_rms'] for g in G)/len(G):.2f}  n_window "
          f"{sum(g['n_window'] for g in G)/len(G):.1f}  b/T "
          f"{sum(g['b_over_t'] for g in G)/len(G):.2f}  Pi max "
          f"{max(g['pi_surr'] for g in G):.3g}  n_eff min "
          f"{min(g['n_eff'] for g in G):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
