"""Monte Carlo anatomy of the laplace_policy selection gradient on a fixed batch.

On one fixed checkpoint and one fixed training micro-batch, for M independent
noise draws m (each draw = one sampled support per gate row), computes per
parameter group (embedding, and per block: branches, encoder, decoder):

  g_value        the clean Top-K CE gradient (sampling off): the reference
  g_val^(m)      the CE gradient through the SAMPLED hard masks (no selection term)
  g_lr^(m)       the likelihood-ratio selection term alone, gamma = 1, with the
                 baseline B (--baseline: the checkpoint's EMA value, or "clean":
                 the batch's clean CE)
  g_rb^(m)       the Rao-Blackwell selection term (total minus g_val^(m)), on the
                 SAME draw (same noise generator seed, so the same supports)
  g_rbfo^(m)     the same with policy_rb_scope=first_order

and reports, per group and estimator: the norm of the Monte Carlo mean
|mean_m g|, the per-draw noise sqrt(tr Cov) = sqrt(sum_m |g - mean|^2 / (M-1)),
the bias-corrected signal |E g| = sqrt(max(0, |mean|^2 - tr Cov / M)), the
per-draw signal-to-noise ratio |E g| / sqrt(tr Cov), the cosine of the mean
with g_value, and the median per-draw norm.  Also gamma* = |g_value| /
median_m |g_lr^(m)| (whole model).  Writes everything to --json.

    python scripts/policy_grad_probe.py --ckpt RUN/ckpt_step2000.pt --draws 64 --json out.json
    python scripts/policy_grad_probe.py --config CFG.yaml [--weights OTHER.pt] ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.bottleneck.laplace_policy import (PolicyForwardSettings,  # noqa: E402
                                               make_generator, policy_gates)
from wsparse.bottleneck.rblapsum import first_order_backward  # noqa: E402
from wsparse.config import config_from_dict, load_config  # noqa: E402
from wsparse.data import build_streams, load_meta  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.utils import autocast_context, resolve_dtype, set_seed  # noqa: E402


def group_of(name: str) -> str:
    m = re.match(r"blocks\.(\d+)\.(.*)", name)
    if not m:
        return "embed/other"
    b, rest = m.group(1), m.group(2)
    if "bottleneck" in rest or "gate" in rest:
        if "in_proj" in rest or "enc" in rest:
            return f"block{b}.encoder"
        if "out_proj" in rest or "dec" in rest:
            return f"block{b}.decoder"
        return f"block{b}.bottleneck_other"
    return f"block{b}.branches"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--weights", default="", help="state dict to load into --config's model")
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--micro", type=int, default=0)
    ap.add_argument("--baseline", default="ema", help="ema (checkpoint) | clean | <float>")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--estimators", default="lr,rb,rbfo")
    ap.add_argument("--json", default="")
    args, unknown = ap.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    payload = None
    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tree = payload["config"]
        if unknown:
            from wsparse.config import apply_overrides
            tree = apply_overrides(tree, list(unknown))
        cfg = config_from_dict(tree)
    else:
        cfg = load_config(args.config, list(unknown))
    meta = load_meta(cfg.data.data_dir)
    cfg.model.vocab_size = int(meta["vocab_size"])
    set_seed(cfg.train.seed)
    model = build_model(cfg.model)
    ctl = apply_activation_bottleneck(model, cfg.activation_bottleneck, max_steps=cfg.train.max_steps)
    step = 0
    if payload is not None:
        model.load_state_dict(payload["model"])
        step = int(payload["step"])
    elif args.weights:
        w = torch.load(args.weights, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(w["model"], strict=False)
        step = int(w.get("step", 0))
        print(f"[probe] weights {args.weights} (step {step}); missing {missing} unexpected {unexpected}")
    elif cfg.model.decouple:
        from wsparse.decouple import md_init_
        md_init_(model, cfg.model.decouple_gains)
    model.to(device)
    model.train()
    tau = ctl.set_step(max(0, step - 1))
    dtype = resolve_dtype(cfg.train.dtype, device)
    gates = policy_gates(model)
    bn = cfg.activation_bottleneck
    micro = args.micro or cfg.train.micro_batch_size
    train_stream, _ = build_streams(cfg.data, seed=args.seed)
    x, y = train_stream.batch(micro, device)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in params]
    plist = [p for _, p in params]
    groups = sorted({group_of(n) for n in names}, key=lambda g: (g.split(".")[0], g))
    anchor = model.tok_emb.weight
    print(f"[probe] {cfg.train.run_name} step {step} K={bn.k} J={bn.j} "
          f"{bn.policy_temperature_mode}/{bn.policy_width_gradient} tau={tau:g} micro={micro} "
          f"draws={args.draws} groups={len(groups)}")

    def grads_of(scalar, retain=True):
        gs = torch.autograd.grad(scalar, plist, retain_graph=retain, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach().float() for g, p in zip(gs, plist)]

    def set_estimator(est, scope="full"):
        for gte in gates:
            gte.policy_estimator = est
            gte.policy_rb_scope = scope

    # clean value gradient (sampling off)
    set_estimator("likelihood_ratio")
    with autocast_context(device, dtype):
        out = model(x, y, return_loss_details=True, policy=PolicyForwardSettings(sample=False))
    clean_ce = float(out.ce.detach())
    g_value = grads_of(out.ce, retain=False)
    del out
    if args.baseline == "ema":
        B = float(payload["policy_state"]["baseline"]) if payload and "policy_state" in payload \
            else math.log(cfg.model.vocab_size)
    elif args.baseline == "clean":
        B = clean_ce
    else:
        B = float(args.baseline)
    print(f"[probe] clean CE {clean_ce:.4f}, baseline B = {B:.4f}")

    ests = ["val"] + [e for e in args.estimators.split(",") if e]
    acc = {e: {"sum": [torch.zeros_like(p, dtype=torch.float32) for p in plist],
               "sq": defaultdict(float), "norms": defaultdict(list), "dots": defaultdict(list),
               "total_norms": []} for e in ests}
    sampled_ce = []

    def accumulate(e, g):
        a = acc[e]
        tot = 0.0
        per = defaultdict(float)
        dot = defaultdict(float)
        for i, (n, t) in enumerate(zip(names, g)):
            a["sum"][i] += t
            gname = group_of(n)
            nn2 = float(t.pow(2).sum())
            per[gname] += nn2
            dot[gname] += float((t * g_value[i]).sum())
            tot += nn2
        for gname in groups:
            a["sq"][gname] += per[gname]
            a["norms"][gname].append(math.sqrt(per[gname]))
            a["dots"][gname].append(dot[gname])
        a["total_norms"].append(math.sqrt(tot))

    for m in range(args.draws):
        seed = 10_000 + m
        # likelihood-ratio forward: value path and the LR selection term
        set_estimator("likelihood_ratio")
        settings = PolicyForwardSettings(sample=True, generator=make_generator(device, seed))
        with autocast_context(device, dtype):
            out = model(x, y, return_loss_details=True, policy=settings)
        sampled_ce.append(float(out.ce.detach()))
        g_val = grads_of(out.ce, retain=True)
        accumulate("val", g_val)
        if "lr" in ests:
            w = out.seq_valid.float() / out.seq_valid.sum().float()
            sup = (w * (out.seq_ce.detach() - B) * out.policy_log_prob.float()).sum()
            accumulate("lr", grads_of(sup, retain=False))
        del out
        for e, scope in (("rb", "full"), ("rbfo", "first_order")):
            if e not in ests:
                continue
            set_estimator("rao_blackwell", scope)
            settings = PolicyForwardSettings(sample=True, generator=make_generator(device, seed))
            with autocast_context(device, dtype):
                out = model(x, y, return_loss_details=True, policy=settings)
            model.zero_grad(set_to_none=True)
            if scope == "first_order":
                first_order_backward(out.ce, anchor)
            else:
                out.ce.backward()
            g_tot = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.detach().float()
                     for p in plist]
            accumulate(e, [a - b for a, b in zip(g_tot, g_val)])
            model.zero_grad(set_to_none=True)
            del out
        del g_val
        if (m + 1) % 16 == 0:
            print(f"[probe] {m + 1}/{args.draws} draws")
    set_estimator(bn.policy_estimator, bn.policy_rb_scope)

    # ---- statistics ---------------------------------------------------------- #
    M = args.draws
    gv_norm = {gname: 0.0 for gname in groups}
    for n, t in zip(names, g_value):
        gv_norm[group_of(n)] += float(t.pow(2).sum())
    gv_norm = {k: math.sqrt(v) for k, v in gv_norm.items()}
    report = {"run": cfg.train.run_name, "step": step, "k": bn.k, "j": bn.j, "tau": tau,
              "mode": bn.policy_temperature_mode, "width_gradient": bn.policy_width_gradient,
              "micro": micro, "draws": M, "baseline": B, "clean_ce": clean_ce,
              "sampled_ce_mean": statistics.mean(sampled_ce),
              "g_value_norm": math.sqrt(sum(v * v for v in gv_norm.values())),
              "groups": {}, "total": {}}
    for e in ests:
        a = acc[e]
        mean_sq = defaultdict(float)
        mean_dot = defaultdict(float)
        for n, s, gvt in zip(names, a["sum"], g_value):
            mu = s / M
            gname = group_of(n)
            mean_sq[gname] += float(mu.pow(2).sum())
            mean_dot[gname] += float((mu * gvt).sum())
        tot_mean_sq = sum(mean_sq.values())
        tot_trcov = 0.0
        for gname in groups:
            trcov = max(0.0, (a["sq"][gname] - M * mean_sq[gname]) / (M - 1))
            tot_trcov += trcov
            sig2 = max(0.0, mean_sq[gname] - trcov / M)
            report["groups"].setdefault(gname, {})[e] = {
                "mean_norm": math.sqrt(mean_sq[gname]), "noise": math.sqrt(trcov),
                "signal": math.sqrt(sig2), "snr": math.sqrt(sig2) / max(1e-30, math.sqrt(trcov)),
                "cos_mean_value": mean_dot[gname] / max(1e-30, math.sqrt(mean_sq[gname]) * gv_norm[gname]),
                "median_norm": statistics.median(a["norms"][gname]),
                "g_value_norm": gv_norm[gname]}
        sig2 = max(0.0, tot_mean_sq - tot_trcov / M)
        report["total"][e] = {"mean_norm": math.sqrt(tot_mean_sq), "noise": math.sqrt(tot_trcov),
                              "signal": math.sqrt(sig2),
                              "snr": math.sqrt(sig2) / max(1e-30, math.sqrt(tot_trcov)),
                              "cos_mean_value": sum(mean_dot.values()) / max(
                                  1e-30, math.sqrt(tot_mean_sq) * report["g_value_norm"]),
                              "median_norm": statistics.median(a["total_norms"])}
    if "lr" in ests:
        report["gamma_star"] = report["g_value_norm"] / report["total"]["lr"]["median_norm"]
    # cross-estimator agreement of the means (whole model)
    def cos_means(e1, e2):
        num = sum(float((s1 * s2).sum()) for s1, s2 in zip(acc[e1]["sum"], acc[e2]["sum"]))
        n1 = math.sqrt(sum(float(s.pow(2).sum()) for s in acc[e1]["sum"]))
        n2 = math.sqrt(sum(float(s.pow(2).sum()) for s in acc[e2]["sum"]))
        return num / max(1e-30, n1 * n2)
    for e1, e2 in (("lr", "rb"), ("lr", "rbfo"), ("rb", "rbfo")):
        if e1 in ests and e2 in ests:
            report[f"cos_mean_{e1}_{e2}"] = cos_means(e1, e2)

    print(f"\n|g_value| = {report['g_value_norm']:.4g}   clean CE {clean_ce:.4f}   "
          f"sampled CE {report['sampled_ce_mean']:.4f}   B {B:.4f}")
    print(f"{'whole model':24s} " + "  ".join(
        f"{e}: |mean| {v['mean_norm']:.3g} noise {v['noise']:.3g} signal {v['signal']:.3g} "
        f"snr {v['snr']:.2g} cos {v['cos_mean_value']:+.2f}" for e, v in report["total"].items()))
    if "gamma_star" in report:
        print(f"gamma* = |g_value| / median |g_lr| = {report['gamma_star']:.3g}")
    for k in [k for k in report if k.startswith("cos_mean_")]:
        print(f"{k} = {report[k]:+.3f}")
    print(f"\n{'group':22s} {'|g_value|':>9s} " + " ".join(
        f"{e + ' |mean|':>11s} {e + ' noise':>10s} {e + ' snr':>8s} {e + ' cos':>7s}" for e in ests))
    for gname in groups:
        r = report["groups"][gname]
        print(f"{gname:22s} {r[ests[0]]['g_value_norm']:9.3g} " + " ".join(
            f"{r[e]['mean_norm']:11.3g} {r[e]['noise']:10.3g} {r[e]['snr']:8.2g} "
            f"{r[e]['cos_mean_value']:+7.2f}" for e in ests))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
