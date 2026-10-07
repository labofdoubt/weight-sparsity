"""Figures and tables of the policy_cr note (docs/policy-temperature-code-residual.tex).

Inputs, in docs/figures/policy-cr/data/ unless given otherwise:
  curves_policy.json        scripts/extract_policy_curves.py over the vi_pol_rs_* runs
  calib_abs.json            scripts/policy_calibration_table.py --json, absolute-T probes
  calib_rs.json             the same for the relative_span probes
  stoch_eval.json           scripts/policy_stochastic_eval.py (optional; 8 draws at the end)
  ../kj-cr/data/curves_new.json   the madrid reference runs (RBLapSum T=2, hard Top-K')
  ref/<run>/metrics.jsonl   madrid reference training metrics (optional; gradient norms)

Figures (docs/figures/policy-cr/):
  pc_calib.png          calibration: median exchange over blocks and pool span against
                        step for the absolute probes (reds by T) and the relative_span
                        probes (blues by tau), one column per cell
  pc_val_clean.png      clean deterministic Top-K validation CE, one panel per cell:
                        policy constant / annealed, RBLapSum T=2, hard Top-K (same K)
  pc_delta.png          headline: Delta_P-RB (final and best clean CE) per cell, both arms;
                        right panel Delta_P-hard
  pc_widen_j.png        K=32: clean CE against K+J for policy (both arms) and RBLapSum,
                        and G(K,J) = L(K,J) - L(K,K)
  pc_pairs.png          clean and stochastic validation CE of each pair, one panel per cell
  pc_blocks_exchange.png, pc_blocks_sgrad.png, pc_blocks_span.png
                        per-block exchange fraction, score-gradient RMS and pool span
                        against step, one panel per run (rows cells, columns arms)
  pc_gradnorm.png       gradient norm against step (log) for the eight runs and the
                        RBLapSum / hard references, with the clip threshold

    python scripts/plot_policy_cr.py [--data DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CELLS = [(32, 32), (32, 480), (128, 128), (256, 256)]
C_CONST, C_ANNEAL = "#1f4e9c", "#4fa3d1"
C_RB, C_HARD, C_HARD_POOL = "0.35", "#c0392b", "#e59866"
REDS, BLUES, VIRIDIS = cm.get_cmap("Reds"), cm.get_cmap("Blues"), cm.get_cmap("viridis")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return default


def cell_runs(curves, k, j):
    out = {}
    for name, rec in curves.items():
        c = rec["cfg"]
        if (c["k"], c["j"]) == (k, j):
            out["anneal" if c["schedule"] == "exponential" else "const"] = (name, rec)
    return out


def ref_runs(ref, k, j):
    return ref.get(f"ma_cr_rbk_k{k}_j{j}_t2"), ref.get(f"ma_cr_hard_k{k}_pnorm"), \
        ref.get(f"ma_cr_hard_k{k + j}_pnorm")


def final_best(rec_val):
    """(final clean CE, best clean CE, last step) of a [[step, ce, ...]] list."""
    if not rec_val:
        return None, None, None
    return rec_val[-1][1], min(v[1] for v in rec_val), rec_val[-1][0]


def at_step(rec_val, step):
    for v in rec_val:
        if v[0] == step:
            return v[1]
    return None


def save(fig, outdir, fn):
    fig.tight_layout()
    p = os.path.join(outdir, fn)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(p)


# --------------------------------------------------------------------------- #
def fig_calib(cal_abs, cal_rs, outdir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True)
    for col, (k, j) in enumerate(CELLS):
        ax_q, ax_s = axes[0, col], axes[1, col]
        for cal, cmap, rng, ls, lab in ((cal_abs, REDS, (0.35, 0.95), "-", "absolute $T$"),
                                        (cal_rs, BLUES, (0.5, 0.95), "-", "relative span $\\tau$")):
            runs = sorted((e["T"], e) for e in cal.values() if (e["k"], e["j"]) == (k, j))
            for i, (t, e) in enumerate(runs):
                c = cmap(rng[0] + (rng[1] - rng[0]) * i / max(1, len(runs) - 1))
                steps = sorted(int(s) for s in e["steps"])
                q = [e["steps"][str(s)]["q_median"] for s in steps]
                sp = [e["steps"][str(s)]["span_median"] for s in steps]
                name = ("$T$" if cmap is REDS else "$\\tau$") + f"$={t:g}$"
                ax_q.plot(steps, q, ls, color=c, lw=1.8, label=name)
                ax_s.plot(steps, sp, ls, color=c, lw=1.8)
        ax_q.axhspan(0.15, 0.20, color="0.85", zorder=0)
        ax_q.axhline(0.03, color="0.4", ls=":", lw=1)
        ax_q.axvline(30, color="0.4", ls="--", lw=0.8)
        ax_q.set_title(f"$(K, J) = ({k}, {j})$")
        ax_q.set_ylim(0, 0.42)
        ax_s.set_yscale("log")
        ax_s.set_xlabel("training step")
        ax_q.grid(alpha=0.3)
        ax_s.grid(alpha=0.3)
        ax_q.legend(fontsize=7, ncol=2, loc="upper right")
    axes[0, 0].set_ylabel("median over blocks of exchange $q_\\ell$")
    axes[1, 0].set_ylabel("pool span $s_{(K+1)} - s_{(K+J)}$\n(median over blocks)")
    save(fig, outdir, "pc_calib.png")


def fig_val_clean(curves, ref, outdir, ycap=None):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (k, j) in zip(axes, CELLS):
        runs = cell_runs(curves, k, j)
        rb, hard, hard_pool = ref_runs(ref, k, j)
        lo, hi = math.inf, -math.inf
        for arm, color in (("const", C_CONST), ("anneal", C_ANNEAL)):
            if arm in runs:
                v = runs[arm][1]["val"]
                ax.plot([p[0] for p in v], [p[1] for p in v], "-", color=color, lw=2.0,
                        label=f"policy, {'constant' if arm == 'const' else 'annealed'} $\\tau$")
                lo = min(lo, min(p[1] for p in v))
                hi = max(hi, max(p[1] for p in v[len(v) // 3:]) if len(v) > 3 else v[-1][1])
        for rec, color, ls, lab in ((rb, C_RB, ":", "RBLapSum $T=2$"),
                                    (hard, C_HARD, "--", f"hard Top-${k}$"),
                                    (hard_pool, C_HARD_POOL, "--", f"hard Top-${k + j}$")):
            if rec:
                ax.plot([p[0] for p in rec["val"]], [p[1] for p in rec["val"]], ls, color=color,
                        lw=1.8, label=lab)
                lo = min(lo, min(p[1] for p in rec["val"]))
        top = ycap if ycap else max(hi + 0.05, lo + 0.6)
        ax.set_ylim(lo - 0.02, top)
        ax.set_xlim(0, 20000)
        ax.set_title(f"$(K, J) = ({k}, {j})$")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("clean Top-$K$ validation CE (nats)")
    save(fig, outdir, "pc_val_clean.png")


def fig_delta(curves, ref, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = range(len(CELLS))
    for ax, which in zip(axes, ("rb", "hard")):
        for i, (k, j) in enumerate(CELLS):
            runs = cell_runs(curves, k, j)
            rb, hard, _ = ref_runs(ref, k, j)
            base = rb if which == "rb" else hard
            if not base:
                continue
            b_final, b_best = base["summary"]["val/ce"], base["summary"]["best_val_ce"]
            for arm, color, dx in (("const", C_CONST, -0.12), ("anneal", C_ANNEAL, 0.12)):
                if arm not in runs:
                    continue
                f, b, s = final_best(runs[arm][1]["val"])
                if f is None:
                    continue
                if s != 20000:      # partial run: compare at the same step
                    b_final = at_step(base["val"], s) or b_final
                    b_best = min(v[1] for v in base["val"] if v[0] <= s)
                ax.plot(i + dx, f - b_final, "o", color=color, ms=8)
                ax.plot(i + dx, b - b_best, "o", mfc="none", mec=color, mew=1.8, ms=11)
        ax.axhline(0, color="0.2", lw=1)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"({k}, {j})" for k, j in CELLS])
        ax.set_xlabel("$(K, J)$")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("$\\Delta_{\\mathrm{P-RB}} = L^{\\mathrm{policy}}_{\\mathrm{clean}}"
                       " - L^{\\mathrm{RBLapSum}}_{\\mathrm{clean}}$ (nats)")
    axes[1].set_ylabel("$\\Delta_{\\mathrm{P-hard}} = L^{\\mathrm{policy}}_{\\mathrm{clean}}"
                       " - L^{\\mathrm{hard\\,Top}\\text{-}K}_{\\mathrm{clean}}$ (nats)")
    hd = [Line2D([0], [0], marker="o", ls="", color=C_CONST, ms=8, label="constant, final"),
          Line2D([0], [0], marker="o", ls="", mfc="none", mec=C_CONST, mew=1.8, ms=11,
                 label="constant, best"),
          Line2D([0], [0], marker="o", ls="", color=C_ANNEAL, ms=8, label="annealed, final"),
          Line2D([0], [0], marker="o", ls="", mfc="none", mec=C_ANNEAL, mew=1.8, ms=11,
                 label="annealed, best")]
    axes[0].legend(handles=hd, fontsize=8)
    axes[0].set_title("policy minus RBLapSum $T=2$ (same $K$, $J$)")
    axes[1].set_title("policy minus hard Top-$K$ (same active count)")
    save(fig, outdir, "pc_delta.png")


def fig_widen_j(curves, ref, outdir):
    k = 32
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for arm, color, lab in (("const", C_CONST, "policy, constant"),
                            ("anneal", C_ANNEAL, "policy, annealed"), ("rb", C_RB, "RBLapSum $T=2$")):
        xs, ys, bs = [], [], []
        for j in (32, 480):
            if arm == "rb":
                rec = ref.get(f"ma_cr_rbk_k{k}_j{j}_t2")
                if not rec:
                    continue
                f, b = rec["summary"]["val/ce"], rec["summary"]["best_val_ce"]
            else:
                runs = cell_runs(curves, k, j)
                if arm not in runs:
                    continue
                f, b, _ = final_best(runs[arm][1]["val"])
            xs.append(k + j)
            ys.append(f)
            bs.append(b)
        if len(xs) == 2:
            axes[0].plot(xs, ys, "o-", color=color, lw=2, ms=7, label=lab + ", final")
            axes[0].plot(xs, bs, "o:", color=color, lw=1.4, ms=6, mfc="none", label=lab + ", best")
            axes[1].bar({"const": -0.25, "anneal": 0.0, "rb": 0.25}[arm], ys[1] - ys[0], 0.22,
                        color=color, label=lab)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([64, 512])
    axes[0].set_xticklabels(["$K+J=64$\n$(J=32)$", "$K+J=512$\n$(J=480)$"])
    axes[0].set_ylabel("clean validation CE (nats)")
    axes[0].set_title("$K = 32$: widening the candidate window")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="0.2", lw=1)
    axes[1].set_xticks([])
    axes[1].set_ylabel("$G(32, 480) = L(32, 480) - L(32, 32)$ (final, nats)")
    axes[1].set_title("gain from $J = 32 \\to 480$ (negative: wider is better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    save(fig, outdir, "pc_widen_j.png")


def fig_pairs(curves, outdir, stoch=None):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (k, j) in zip(axes, CELLS):
        runs = cell_runs(curves, k, j)
        lo, hi = math.inf, -math.inf
        for arm, color in (("const", C_CONST), ("anneal", C_ANNEAL)):
            if arm not in runs:
                continue
            v = runs[arm][1]["val"]
            s = [p[0] for p in v]
            lab = "constant" if arm == "const" else "annealed"
            ax.plot(s, [p[1] for p in v], "-", color=color, lw=2.0, label=f"{lab}, clean")
            ax.plot(s, [p[2] for p in v], "--", color=color, lw=1.5, label=f"{lab}, stochastic")
            tail = v[len(v) // 3:] if len(v) > 3 else v
            lo = min(lo, min(min(p[1], p[2]) for p in v))
            hi = max(hi, max(max(p[1], p[2]) for p in tail))
            if stoch and runs[arm][0] in stoch:
                e = stoch[runs[arm][0]]
                ax.errorbar([e["step"] + 150], [e["stochastic_ce"]], yerr=[e["stochastic_std"]],
                            fmt="s", color=color, ms=5, capsize=3)
        ax.set_ylim(lo - 0.02, max(hi + 0.05, lo + 0.4))
        ax.set_xlim(0, 20400)
        ax.set_title(f"$(K, J) = ({k}, {j})$")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("validation CE (nats)")
    save(fig, outdir, "pc_pairs.png")


def fig_blocks(curves, outdir, key, fn, ylabel, log=False):
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 13), sharex=True)
    for r, (k, j) in enumerate(CELLS):
        runs = cell_runs(curves, k, j)
        for c, arm in enumerate(("const", "anneal")):
            ax = axes[r, c]
            if arm not in runs:
                ax.axis("off")
                continue
            series = runs[arm][1]["blocks"].get(key, [])
            if not series:
                continue
            steps = [p[0] for p in series]
            nb = len(series[0][1])
            for b in range(nb):
                ax.plot(steps, [p[1][b] for p in series], color=VIRIDIS(b / max(1, nb - 1)),
                        lw=1.0, label=f"block {b}")
            if log:
                ax.set_yscale("log")
            if key == "exchange":
                ax.axhspan(0.15, 0.20, color="0.88", zorder=0)
                ax.axhline(0.03, color="0.4", ls=":", lw=1)
            ax.set_title(f"$({k}, {j})$, {'constant' if arm == 'const' else 'annealed'}"
                         f" $\\tau_0 = {runs[arm][1]['cfg']['tau0']:g}$", fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(ylabel)
    axes[0, 1].legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    save(fig, outdir, fn)


def fig_gradnorm(curves, refdir, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for name, rec in sorted(curves.items()):
        c = rec["cfg"]
        color = {(32, 32): "#1b9e77", (32, 480): "#d95f02", (128, 128): "#7570b3",
                 (256, 256): "#e7298a"}[(c["k"], c["j"])]
        ls = "-" if c["schedule"] == "constant" else "--"
        tr = rec["train"]
        axes[0].plot([p[0] for p in tr], [p[3] for p in tr], ls, color=color, lw=1.0,
                     label=f"({c['k']}, {c['j']}) {'const' if ls == '-' else 'anneal'}")
        cl = rec["clip"]
        axes[1].plot([p[0] for p in cl], [p[2] for p in cl], ls, color=color, lw=1.4)
    for run, color in (("ma_cr_rbk_k32_j480_t2", "0.3"), ("ma_cr_hard_k32_pnorm", C_HARD)):
        path = os.path.join(refdir, run, "metrics.jsonl")
        if os.path.isfile(path):
            rows = [json.loads(line) for line in open(path)]
            tr = [(r["step"], r["train/grad_norm"]) for r in rows if "train/grad_norm" in r]
            axes[0].plot([s for s, _ in tr], [g for _, g in tr], ":", color=color, lw=1.0,
                         label=run.replace("ma_cr_", "").replace("_", " "))
    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", lw=1, ls="-.")
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
    axes[0].set_ylabel("gradient norm before clipping (logged steps)")
    axes[1].set_ylabel("median gradient norm per 500-step window")
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, outdir, "pc_gradnorm.png")


# --------------------------------------------------------------------------- #
def tables(curves, ref, cal_abs, cal_rs, stoch):
    print("\n% calibration: median exchange over blocks at step 30 (max block)")
    for name, cal in (("absolute", cal_abs), ("relative_span", cal_rs)):
        for (k, j) in CELLS:
            row = []
            for t, e in sorted((e["T"], e) for e in cal.values() if (e["k"], e["j"]) == (k, j)):
                s30 = e["steps"].get("30")
                s200 = e["steps"].get("200")
                row.append(f"{t:g}: {s30['q_median']:.3f} ({s30['q_max']:.3f})"
                           + (f" -> {s200['q_median']:.3f} @200" if s200 else ""))
            print(f"%  {name:13s} ({k},{j}): " + "; ".join(row))
    print("\n% final / best clean CE, Delta vs RBLapSum T=2 and vs hard Top-K")
    for (k, j) in CELLS:
        runs = cell_runs(curves, k, j)
        rb, hard, hard_pool = ref_runs(ref, k, j)
        for arm in ("const", "anneal"):
            if arm not in runs:
                continue
            f, b, s = final_best(runs[arm][1]["val"])
            rbf, rbb = rb["summary"]["val/ce"], rb["summary"]["best_val_ce"]
            hf, hb = hard["summary"]["val/ce"], hard["summary"]["best_val_ce"]
            if s != 20000:
                rbf, hf = at_step(rb["val"], s), at_step(hard["val"], s)
                rbb = min(v[1] for v in rb["val"] if v[0] <= s)
                hb = min(v[1] for v in hard["val"] if v[0] <= s)
            sto = runs[arm][1]["val"][-1][2]
            print(f"({k:3d},{j:3d}) {arm:6s} step {s:5d} final {f:.4f} best {b:.4f} sto {sto:.4f} | "
                  f"RB {rbf:.4f}/{rbb:.4f} dPRB {f - rbf:+.4f}/{b - rbb:+.4f} | hard{k} "
                  f"{hf:.4f} dPhard {f - hf:+.4f}/{b - hb:+.4f}"
                  + (f" | hard{k + j} {at_step(hard_pool['val'], s) or hard_pool['summary']['val/ce']:.4f}"
                     if hard_pool else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
    ap.add_argument("--data", default=os.path.join(here, "policy-cr", "data"))
    ap.add_argument("--out", default=os.path.join(here, "policy-cr"))
    ap.add_argument("--ref", default=os.path.join(here, "kj-cr", "data", "curves_new.json"))
    args = ap.parse_args()
    curves = load(os.path.join(args.data, "curves_policy.json"), {})
    cal_abs = load(os.path.join(args.data, "calib_abs.json"), {})
    cal_rs = load(os.path.join(args.data, "calib_rs.json"), {})
    stoch = load(os.path.join(args.data, "stoch_eval.json"))
    ref = load(args.ref, {})
    os.makedirs(args.out, exist_ok=True)
    fig_calib(cal_abs, cal_rs, args.out)
    fig_val_clean(curves, ref, args.out)
    fig_delta(curves, ref, args.out)
    fig_widen_j(curves, ref, args.out)
    fig_pairs(curves, args.out, stoch)
    fig_blocks(curves, args.out, "exchange", "pc_blocks_exchange.png", "exchange $q_\\ell$")
    fig_blocks(curves, args.out, "score_grad_rms", "pc_blocks_sgrad.png",
               "score-gradient RMS $\\gamma\\,|\\mathrm{sign}(r-u)|/T$", log=True)
    fig_blocks(curves, args.out, "span", "pc_blocks_span.png",
               "pool span $s_{(K+1)} - s_{(K+J)}$", log=True)
    fig_gradnorm(curves, os.path.join(args.data, "ref"), args.out)
    tables(curves, ref, cal_abs, cal_rs, stoch)


if __name__ == "__main__":
    main()
