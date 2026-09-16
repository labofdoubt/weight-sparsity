"""Forensics for the RBLapSum ``through_rank_kappa`` stability investigation.

Three subcommands, matched to the three evidence sources:

``tb``      Dump the training-time scalar diagnostics (loss, val CE, rank-
            boundary level, support-gradient norm, cap fraction ...) from the
            TensorBoard event files of the 20k campaign runs, detect the
            destabilisation onset step per run, and write a fate table.

``ladder``  Measure the *score geometry* at every checkpoint of the ladder
            datasets (the [ckpt, layer, batch, pos, feature] signed-z arrays):
            boundary level s_(K+1), the K-gap, the local rank spacing
            delta(b) = (s_(K-m) - s_(K+m)) / 2m, the kernel mass Sum(kappa),
            the participation count n_eff = (Sum k)^2 / Sum k^2, and activation
            scales.  Every run is also measured at the *counterfactual*
            boundaries K=32 and K=64 so the k-dependence of the geometry can be
            compared inside one and the same model.

``probe``   Reconstruct the actual surrogate force from the early-training
            probe captures (signed score z, upstream u = dL/d~z, and dL/dz):
            a_i = u_i z_i kappa_i,  g_s = a - q * sum(a),  q = kappa/sum kappa,
            validate the reconstruction against the captured g_z, and measure
            per-step: the boundary kick size, the band-stretch force
            Cov_q(d, w) with w = -u z, the top-K support churn between probes,
            and whether kicked-up members keep their gains (the ratchet test).

All quantities are per-layer; tokens are pooled as means unless noted.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

B0 = 0.1  # rblapsum_boundary_floor used by every kappa run


# ---------------------------------------------------------------- run naming
def parse_run(name: str) -> dict:
    m = re.search(r"kappa_k(\d+)_j(\d+)(?:_t(\d+))?_md_abs", name)
    if not m:
        return {}
    tcode = m.group(3)
    T = {None: 1.0, "2": 2.0, "05": 0.5, "01": 0.1, "15": 1.5, "125": 1.25,
         "075": 0.75, "06": 0.6, "065": 0.65, "085": 0.85, "175": 1.75,
         "3": 3.0, "4": 4.0}.get(tcode, float(tcode) if tcode else 1.0)
    return {"k": int(m.group(1)), "j": int(m.group(2)), "T": T}


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path)


# ------------------------------------------------------------------- tb dump
TB_TAGS = [
    "val/ce", "train/loss", "train/grad_norm",
    "bottleneck/rb_boundary", "bottleneck/rb_b_rank",
    "bottleneck/rb_support_grad_norm", "bottleneck/rb_common_mode",
    "bottleneck/rb_common_mode_raw", "bottleneck/rb_cap_active_frac",
    "bottleneck/in_window_frac", "bottleneck/active_count",
    "bottleneck/rb_temp", "bottleneck/rb_chi", "bottleneck/rb_win_count",
    "bottleneck/rb_kick",
]


def cmd_tb(args) -> None:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    runs = sorted(glob.glob(os.path.join(args.runs_dir, args.glob)))
    summary = {}
    for rd in runs:
        name = os.path.basename(rd)
        tbdir = os.path.join(rd, "tb")
        if not os.path.isdir(tbdir):
            continue
        acc = EventAccumulator(tbdir, size_guidance={"scalars": 0})
        acc.Reload()
        avail = acc.Tags()["scalars"]
        out = {}
        for tag in TB_TAGS:
            if tag not in avail:
                continue
            ev = acc.Scalars(tag)
            out[tag] = {"steps": [e.step for e in ev],
                        "values": [e.value for e in ev]}
        if name == os.path.basename(runs[0]):
            print("available tags:", sorted(avail))
        # onset: first step whose train loss sits 1 nat above the best so far
        onset = None
        tl = out.get("train/loss")
        if tl:
            best = math.inf
            for s, v in zip(tl["steps"], tl["values"]):
                if v < best:
                    best = v
                if s > 100 and v > best + 1.0:
                    onset = s
                    break
        vc = out.get("val/ce")
        summary[name] = {**parse_run(name), "onset_step": onset,
                         "final_val_ce": vc["values"][-1] if vc else None,
                         "min_train_loss": best if tl else None}
        _save_json(os.path.join(args.out_dir, f"tb_{name}.json"), out)
    _save_json(os.path.join(args.out_dir, "tb_summary.json"), summary)
    for n, s in sorted(summary.items()):
        print(f"{n:55s} k={s.get('k')} T={s.get('T')} "
              f"onset={s.get('onset_step')} final={s.get('final_val_ce')}")


# ------------------------------------------------------------ ladder geometry
def geom_at_rank(ss: np.ndarray, K: int, m: int = 4) -> dict:
    """Boundary geometry for an already sorted-descending score array
    ss [tokens, features] at active count K (boundary = rank K+1)."""
    sK = ss[:, K - 1]
    sK1 = ss[:, K]
    gap = sK - sK1
    delta = (ss[:, K - 1 - m] - ss[:, K - 1 + m]) / (2 * m)
    return {"sK": float(sK.mean()), "sK1": float(sK1.mean()),
            "gap": float(gap.mean()),
            "delta": float(delta.mean()),
            "delta_med": float(np.median(delta))}


def kernel_stats(ss: np.ndarray, K: int, J: int, T: float) -> dict:
    b = np.maximum(B0, ss[:, K])[:, None]
    sc = ss[:, :K + J]
    d = sc - b
    kap = np.exp(-np.abs(d) / T) / (2 * T)
    Sk = kap.sum(1)
    Sk2 = (kap ** 2).sum(1)
    n_eff = (Sk ** 2) / np.maximum(Sk2, 1e-30)
    win = (np.abs(d) < T).mean(1)
    cap = (ss[:, K] > B0).mean()
    # exact surrogate gain Pi = ||L_kappa D_z||_F^2 per token via the
    # diagonal-minus-rank-one structure (see analysis/scale_dynamics.py)
    Z = np.maximum(Sk, 1e-30)[:, None]
    colj = (sc * kap) ** 2 * ((1 - kap / Z) ** 2
                              + (Sk2[:, None] - kap ** 2) / (Z ** 2))
    Pi = colj.sum(1)
    return {"sum_kappa": float(Sk.mean()), "n_eff": float(n_eff.mean()),
            "n_eff_med": float(np.median(n_eff)),
            "in_window_frac": float(win.mean()), "cap_frac": float(cap),
            "Pi": float(Pi.mean()), "Pi_med": float(np.median(Pi))}


def cmd_ladder(args) -> None:
    metas = sorted(glob.glob(os.path.join(args.scores_dir, args.glob + ".json")))
    all_out = {}
    for mp in metas:
        meta = json.load(open(mp))
        name = meta["run"]
        info = parse_run(name)
        K, J = meta["k"], meta["j"]
        T = float(meta.get("rblapsum_temperature", info.get("T", 1.0)))
        arr = np.load(mp[:-5] + ".npy", mmap_mode="r")  # [C, L, B, P, F]
        C, L = arr.shape[0], arr.shape[1]
        steps = meta["steps"]
        run_out = {"k": K, "j": J, "T": T, "steps": steps,
                   "batch_ce": meta.get("batch_ce"), "layers": []}
        for li in range(L):
            per_ckpt = []
            for c in range(C):
                z = np.asarray(arr[c, li], dtype=np.float32)
                s = np.abs(z.reshape(-1, z.shape[-1]))
                ss = -np.sort(-s, axis=1)
                rec = {"step": steps[c],
                       "own": geom_at_rank(ss, K),
                       "k32": geom_at_rank(ss, 32),
                       "k64": geom_at_rank(ss, 64),
                       "kern": kernel_stats(ss, K, J, T),
                       "act_top_mean": float(ss[:, :K].mean()),
                       "z_p999": float(np.quantile(s, 0.999)),
                       "z_max": float(s.max())}
                per_ckpt.append(rec)
            run_out["layers"].append(per_ckpt)
        all_out[name] = run_out
        g2 = run_out["layers"][args.ref_layer][0]
        print(f"{name:55s} ck0: sK1={g2['own']['sK1']:.3f} "
              f"gap={g2['own']['gap']:.4f} delta={g2['own']['delta']:.5f} "
              f"neff={g2['kern']['n_eff']:.1f} 1/(2Td)="
              f"{1 / (2 * T * g2['own']['delta']):.1f}")
    _save_json(os.path.join(args.out_dir, "ladder_geometry.json"), all_out)


# ------------------------------------------------------------- probe forces
def cmd_probe(args) -> None:
    metas = sorted(glob.glob(os.path.join(args.probe_dir, args.glob + ".json")))
    all_out = {}
    for mp in metas:
        meta = json.load(open(mp))
        name = meta["run"]
        K, J = meta["k"], meta["j"]
        T = float(meta.get("rblapsum_temperature", 1.0))
        base = mp[:-5]
        try:
            z_a = np.load(base + ".score.npy", mmap_mode="r")
            u_a = np.load(base + ".g_ztilde.npy", mmap_mode="r")
            g_a = np.load(base + ".g_z.npy", mmap_mode="r")
        except FileNotFoundError as e:
            print("skip", name, e)
            continue
        S, L = z_a.shape[0], z_a.shape[1]
        steps = meta["probe_steps"]
        run_out = {"k": K, "j": J, "T": T, "steps": steps,
                   "probe_ce": meta.get("probe_ce"), "layers": []}
        for li in range(args.layers[0], args.layers[1]):
            per_t = []
            prev_top = None
            prev_s = None
            prev_gs = None
            prev_ord = None
            for t in range(S):
                z = np.asarray(z_a[t, li], np.float32).reshape(-1, z_a.shape[-1])
                u = np.asarray(u_a[t, li], np.float32).reshape(z.shape)
                gz = np.asarray(g_a[t, li], np.float32).reshape(z.shape)
                s = np.abs(z)
                order = np.argsort(-s, axis=1)
                rows = np.arange(s.shape[0])[:, None]
                cand = order[:, :K + J]
                s_c = np.take_along_axis(s, cand, 1)
                z_c = np.take_along_axis(z, cand, 1)
                u_c = np.take_along_axis(u, cand, 1)
                b = np.maximum(B0, s_c[:, K])[:, None]
                dmar = s_c - b
                kap = np.exp(-np.abs(dmar) / T) / (2 * T)
                a = u_c * z_c * kap
                q = kap / np.maximum(kap.sum(1, keepdims=True), 1e-30)
                cap = (s_c[:, K] > B0)[:, None]
                g_s = np.where(cap, a - q * a.sum(1, keepdims=True), a)
                # validate against captured g_z
                mask = np.zeros_like(s_c); mask[:, :K] = 1.0
                gz_pred_c = u_c * mask + np.sign(z_c) * g_s
                gz_c = np.take_along_axis(gz, cand, 1)
                num = np.abs(gz_pred_c - gz_c).mean()
                den = np.abs(gz_c).mean() + 1e-30
                # forces
                w = -u_c * z_c
                inwin = np.abs(dmar) < T
                kick = np.abs(g_s)[inwin].mean() if inwin.any() else 0.0
                wq_mean = (q * w).sum(1, keepdims=True)
                dq_mean = (q * dmar).sum(1, keepdims=True)
                cov_qdw = ((q * (dmar - dq_mean) * (w - wq_mean)).sum(1)).mean()
                Sk = kap.sum(1); Sk2 = (kap ** 2).sum(1)
                rec = {"step": int(steps[t]),
                       "recon_rel_err": float(num / den),
                       "kick_win": float(kick),
                       "gs_rms": float(np.sqrt((g_s ** 2).mean())),
                       "cov_q_d_w": float(cov_qdw),
                       "sum_kappa": float(Sk.mean()),
                       "n_eff": float((Sk ** 2 / np.maximum(Sk2, 1e-30)).mean()),
                       "sK1": float(s_c[:, K].mean()),
                       "gap": float((s_c[:, K - 1] - s_c[:, K]).mean()),
                       "delta": float(((s_c[:, K - 5] - s_c[:, K + 3]) / 8).mean()),
                       "cap_frac": float(cap.mean()),
                       "act_top_mean": float(s_c[:, :K].mean()),
                       "u_rms_win": float(np.sqrt((u_c[inwin] ** 2).mean()))
                       if inwin.any() else 0.0}
                top = order[:, :K]
                if prev_top is not None:
                    same = np.zeros(s.shape[0])
                    for r in range(s.shape[0]):
                        same[r] = len(np.intersect1d(top[r], prev_top[r],
                                                     assume_unique=True)) / K
                    rec["topk_overlap_prev"] = float(same.mean())
                    # ratchet: realised score change of previous window members
                    # vs the kick they received 10 steps earlier
                    ds = np.take_along_axis(s, prev_ord, 1) - \
                        np.take_along_axis(prev_s, prev_ord, 1)
                    sel = np.abs(prev_s_c_d) < T
                    if sel.any():
                        dsw = ds[:, :K + J][sel]
                        kw = -prev_gs[sel]  # descent moves scores along -g_s
                        cc = (np.corrcoef(dsw, kw)[0, 1]
                              if dsw.std() > 0 and kw.std() > 0 else 0.0)
                        rec["ratchet_corr"] = float(cc)
                        up = dsw[kw > 0]; dn = dsw[kw < 0]
                        rec["ds_kicked_up"] = float(up.mean()) if up.size else 0.0
                        rec["ds_kicked_dn"] = float(dn.mean()) if dn.size else 0.0
                per_t.append(rec)
                prev_top = top
                prev_s = s
                prev_ord = order
                prev_gs = g_s
                prev_s_c_d = dmar
            run_out["layers"].append(per_t)
        all_out[name] = run_out
        r0 = run_out["layers"][0]
        print(f"{name:58s} recon_err~{np.mean([r['recon_rel_err'] for r in r0]):.3f} "
              f"kick~{np.mean([r['kick_win'] for r in r0[5:]]):.2e} "
              f"cov~{np.mean([r['cov_q_d_w'] for r in r0[5:]]):.2e}")
    _save_json(os.path.join(args.out_dir, "probe_forces.json"), all_out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tb")
    t.add_argument("--runs-dir", default="/workspace/runs")
    t.add_argument("--glob", default="uk_rout_rblapsum_kappa_*")
    t.add_argument("--out-dir", required=True)
    t.set_defaults(fn=cmd_tb)
    l = sub.add_parser("ladder")
    l.add_argument("--scores-dir", default="/workspace/analysis/scores")
    l.add_argument("--glob", default="uk_rout_rblapsum_kappa_*")
    l.add_argument("--out-dir", required=True)
    l.add_argument("--ref-layer", type=int, default=4)
    l.set_defaults(fn=cmd_ladder)
    p = sub.add_parser("probe")
    p.add_argument("--probe-dir", default="/workspace/analysis/probe")
    p.add_argument("--glob", default="probe_uk_rout_rblapsum_kappa_*")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers", type=int, nargs=2, default=[0, 8])
    p.set_defaults(fn=cmd_probe)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
