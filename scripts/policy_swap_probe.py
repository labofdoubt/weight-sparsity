"""How good is the first-order single-swap difference of the Rao-Blackwell estimator?

The Rao-Blackwell selection gradient of pool member i is
p_T(theta_i - u_i) * DeltaL_i, with DeltaL_i = L(i in, j out) - L(i out, j in) at fixed
noise of every other member and gate, estimated to first order as g_i v_i - g_j v_j
(g = dL/dy at the gate along hard paths, v the signed values).  This probe measures
DeltaL_i exactly.  On one fixed noise draw (a seeded generator, so every gate draws
the same noise in every forward) a forward hook applies the single swap to the gate's
output; the rest of the network, including the later gates' sampled selections, is
recomputed and the summed token CE compared with the unswapped forward.  fp32, TF32 off.

At fixed noise everywhere the exact difference is dominated by discontinuous
re-selections at the later gates (a swap moves every later gate's input), so the
quantity compared is the expectation over the LATER gates' noise: the generator is
re-seeded right after the probed gate (--down draws), while the earlier gates' and the
probed gate's own noise stay fixed.  For each probed gate, a sample of tokens and per
token the members with the largest crossing density p_T(theta_i - u_i) (u, T rebuilt
from the gate's input as the gate computes them), it reports, over the members, the
correlation and regression slope of E_down[exact] on E_down[linear] (plain and
density-weighted), the standard error of E_down[exact], the per-draw spread of the
exact difference, and the same for the per-token sums sum_i p_i DeltaL_i.

    python scripts/policy_swap_probe.py --ckpt RUN/ckpt.pt [--json out.json]
    python scripts/policy_swap_probe.py --config CFG --weights OTHER.pt ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.bottleneck.laplace_policy import (PolicyCollector,  # noqa: E402
                                               PolicyForwardSettings, make_generator,
                                               policy_gates)
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.data import build_streams, load_meta  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.utils import set_seed  # noqa: E402


def weighted_stats(ex, li, w):
    w = w / w.sum()
    me, ml = (w * ex).sum(), (w * li).sum()
    cov = (w * (ex - me) * (li - ml)).sum()
    ve, vl = (w * (ex - me) ** 2).sum(), (w * (li - ml) ** 2).sum()
    return {"corr": float(cov / torch.sqrt(ve * vl).clamp_min(1e-300)),
            "slope_exact_on_linear": float(cov / vl.clamp_min(1e-300)),
            "rel_rmse": float(torch.sqrt((w * (ex - li) ** 2).sum()
                                         / (w * ex ** 2).sum().clamp_min(1e-300))),
            "mean_abs_exact": float((w * ex.abs()).sum()),
            "mean_abs_linear": float((w * li.abs()).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--gates", default="0,3,7")
    ap.add_argument("--rows", type=int, default=24, help="tokens per gate")
    ap.add_argument("--members", type=int, default=8, help="members per token, by density")
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--down", type=int, default=16, help="draws of the later gates' noise")
    ap.add_argument("--json", default="")
    args, unknown = ap.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = None
    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = config_from_dict(payload["config"])
    else:
        cfg = load_config(args.config, list(unknown))
    meta = load_meta(cfg.data.data_dir)
    cfg.model.vocab_size = int(meta["vocab_size"])
    set_seed(cfg.train.seed)
    model = build_model(cfg.model)
    ctl = apply_activation_bottleneck(model, cfg.activation_bottleneck,
                                      max_steps=cfg.train.max_steps)
    step = 0
    if payload is not None:
        model.load_state_dict(payload["model"])
        step = int(payload["step"])
    elif args.weights:
        w = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(w["model"], strict=False)
        step = int(w.get("step", 0))
    model.to(device).float()
    model.train()
    tau = ctl.set_step(max(0, step - 1))
    gates = policy_gates(model)
    for g in gates:
        g.policy_estimator = "likelihood_ratio"   # plain hard-mask graph: g from the CE only
    bn = cfg.activation_bottleneck
    k, j = bn.k, bn.j
    stream, _ = build_streams(cfg.data, seed=args.seed)
    x, y = stream.batch(args.seqs, device)
    gen_seed = 4242
    swap, box = {}, {}

    def pre_hook(mod, inp):
        if box.get("want") is mod:
            box["a"] = inp[0].detach()

    def hook(mod, inp, out):
        if mod is box.get("target"):
            # the later gates draw from a re-seeded generator: their noise is the
            # downstream draw, the earlier gates' and this gate's noise stay fixed
            box["gen"].manual_seed(1_000_003 + int(box["down"]))
        spec = swap.get(id(mod))
        if spec is not None:
            row, add_i, add_v, rem_i = spec
            o = out.clone()
            flat = o.view(-1, o.shape[-1])
            flat[row, add_i] = add_v
            flat[row, rem_i] = 0.0
            return o
        if box.get("want") is mod and torch.is_grad_enabled():
            out.retain_grad()
            box["y"] = out
        return out

    handles = [g.register_forward_pre_hook(pre_hook) for g in gates]
    handles += [g.register_forward_hook(hook) for g in gates]

    def forward(down_seed, keep=False):
        col = PolicyCollector(keep_samples=keep)
        gen = make_generator(device, gen_seed)
        box["gen"], box["down"] = gen, down_seed
        settings = PolicyForwardSettings(sample=True, collector=col, generator=gen)
        return model(x, y, return_loss_details=True, policy=settings), col

    results = {"run": cfg.train.run_name, "step": step, "tau": tau, "k": k, "j": j,
               "mode": bn.policy_temperature_mode, "width_gradient": bn.policy_width_gradient,
               "seqs": args.seqs, "triples": []}
    torch.manual_seed(args.seed)
    D = args.down
    for gi in [int(s) for s in args.gates.split(",")]:
        gate = gates[gi]
        box.clear()
        box["target"] = gate
        base_tok, grads = [], []
        for d in range(D):
            box["want"] = gate
            out, col = forward(d, keep=(d == 0))
            base_tok.append(out.ce_tokens.detach().double())
            out.ce_tokens.sum().backward()           # L = summed token CE
            n = box["y"].shape[-1]
            grads.append(box["y"].grad.detach().double().view(-1, n))
            if d == 0:
                a = box["a"].double().view(-1, n)    # signed pre-mask values (fixed noise)
                rec = [r for r in col.records if r.gate is gate][0]
            model.zero_grad(set_to_none=True)
            box["want"] = None
        r_all = rec.r.double().view(-1, rec.r.shape[-1])
        idx_all = rec.cand_idx.view(-1, rec.cand_idx.shape[-1])
        for row in torch.randperm(r_all.shape[0])[: args.rows].tolist():
            r, idx = r_all[row], idx_all[row]
            v = a[row, idx]
            s_c = v.abs() if bn.selection_mode == "abs_topk" else v
            u = s_c - s_c.mean() if bn.policy_center_scores else s_c
            mode = bn.policy_temperature_mode
            scale = (s_c[k] - s_c[k + j - 1]) if mode == "relative_span" else (
                s_c[k] if mode == "relative_b" else torch.tensor(1.0, dtype=s_c.dtype))
            if bn.policy_width_gradient == "through" and mode != "absolute":
                u, T = u / scale.clamp_min(1e-6), float(tau)
            else:
                T = float(tau) * float(scale)
            order = torch.argsort(r, descending=True)
            jk, jk1 = int(order[k - 1]), int(order[k])
            sel = torch.zeros_like(r, dtype=torch.bool)
            sel[order[:k]] = True
            theta = torch.where(sel, r[jk1], r[jk])
            dens = torch.exp(-(theta - u).abs() / T) / (2 * T)
            for c in torch.argsort(dens, descending=True)[: args.members].tolist():
                partner = jk1 if sel[c] else jk
                ic, ip = int(idx[c]), int(idx[partner])
                add_i, rem_i = (ip, ic) if sel[c] else (ic, ip)
                sign = -1.0 if sel[c] else 1.0          # DeltaL_i = L(i in) - L(i out)
                ex_d, li_d = [], []
                for d in range(D):
                    swap[id(gate)] = (row, add_i, float(a[row, add_i]), rem_i)
                    with torch.no_grad():
                        out2, _ = forward(d)
                    del swap[id(gate)]
                    ex_d.append(sign * float((out2.ce_tokens.detach().double() - base_tok[d]).sum()))
                    g = grads[d]
                    li_d.append(sign * float(g[row, add_i] * a[row, add_i] - g[row, rem_i] * a[row, rem_i]))
                ext = torch.tensor(ex_d, dtype=torch.float64)
                lit = torch.tensor(li_d, dtype=torch.float64)
                results["triples"].append({
                    "gate": gi, "row": row, "member": c, "selected": bool(sel[c]),
                    "density": float(dens[c]), "T": T,
                    "exact": float(ext.mean()), "exact_sem": float(ext.std() / math.sqrt(D)),
                    "exact_draw_std": float(ext.std()), "linear": float(lit.mean()),
                    "linear_draw_std": float(lit.std()), "exact_draws": ex_d})
        print(f"[swap] gate {gi}: {args.rows} tokens x {args.members} members x {D} draws")
    for h in handles:
        h.remove()
    tr = results["triples"]
    ex = torch.tensor([t["exact"] for t in tr], dtype=torch.float64)
    li = torch.tensor([t["linear"] for t in tr], dtype=torch.float64)
    p = torch.tensor([t["density"] for t in tr], dtype=torch.float64)
    sem = torch.tensor([t["exact_sem"] for t in tr], dtype=torch.float64)
    results["median_exact_sem"] = float(sem.median())
    results["median_exact_draw_std"] = float(torch.tensor([t["exact_draw_std"] for t in tr]).median())
    results["median_abs_exact"] = float(ex.abs().median())
    results["median_abs_linear"] = float(li.abs().median())
    results["share_within_2sem"] = float(((ex - li).abs() <= 2 * sem).double().mean())
    results["plain"] = weighted_stats(ex, li, torch.ones_like(ex))
    results["density_weighted"] = weighted_stats(ex, li, p)
    # per-token Rao-Blackwell sums over the probed members
    rows = {}
    for t in tr:
        key = (t["gate"], t["row"])
        e, l_ = rows.get(key, (0.0, 0.0))
        rows[key] = (e + t["density"] * t["exact"], l_ + t["density"] * t["linear"])
    se = torch.tensor([v[0] for v in rows.values()], dtype=torch.float64)
    sl = torch.tensor([v[1] for v in rows.values()], dtype=torch.float64)
    results["row_sums"] = weighted_stats(se, sl, torch.ones_like(se))
    results["row_sums"]["sign_agreement"] = float(((se * sl) > 0).double().mean())
    for gi in sorted({t["gate"] for t in tr}):
        sub = [i for i, t in enumerate(tr) if t["gate"] == gi]
        results[f"gate{gi}"] = weighted_stats(ex[sub], li[sub], p[sub])
    print(json.dumps({key: val for key, val in results.items() if key != "triples"}, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
