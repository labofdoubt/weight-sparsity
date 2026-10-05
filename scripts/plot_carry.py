"""Figures for docs/carry-surrogate.tex (the carry-aware RBLapSum scopes).

Reads docs/figures/carry/data: analysis/surrogate_probe.py JSONs at
initialization (init/), analysis/code_runs.py JSONs (runs/), the validation
curves of the new screens (curves_bergen.json) and of the code-carried K+J
grid (../kj-cr/data/curves_new.json); writes docs/figures/carry/fig_*.{pdf,png}.

  python scripts/plot_carry.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "carry", "data")
KJ = os.path.join(ROOT, "docs", "figures", "kj-cr", "data")
OUT = os.path.join(ROOT, "docs", "figures", "carry")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GRID = "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.0,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 3.0, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# scope -> (label, color, linestyle)
STYLE = {
    "hard": ("hard Top-$K$", INK, "-"),
    "sw_uni": ("stochastic width, uniform", "#1f77b4", "-"),
    "sw_two": ("stochastic width, two-point", "#17becf", "-"),
    "sw_geo": ("stochastic width, geometric $J/4$", "#9467bd", "-"),
    "sw_geo16": ("stochastic width, geometric $J/16$", "#c5b0d5", "-"),
    "sw_two25": ("stochastic width, two-point $p{=}0.25$", "#9edae5", "-"),
    "ste": ("soft straight-through", "#2ca02c", "-"),
    "inact": ("RBLapSum, candidates only", "#d62728", "-"),
    "cm05": (r"carry, mixed, $\rho{=}0.5$", AQUA, ":"),
    "pool": ("RBLapSum, pool", ORANGE, "-"),
    "fo": ("first order", YELLOW, "-"),
    "cl": ("carry, local", BLUE, "-"),
    "cm": ("carry, mixed", AQUA, "-"),
    "cp": ("carry, persistent", MAGENTA, "--"),
    "ch": ("carry, hard", "#7a5cc6", "--"),
}


def load(*parts):
    p = os.path.join(*parts)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: the backward along the carried code at initialization
# --------------------------------------------------------------------------- #
def fig_init():
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.75), gridspec_kw={"wspace": 0.4})
    for ax, (kj, title) in zip(axes, [("k32_j224", r"a  $K{=}32$, $J{=}224$"),
                                      ("k256_j256", r"b  $K{=}256$, $J{=}256$")]):
        hard = load(DATA, "init", f"init_{kj}_hard.json")
        h = [g["code_grad_rms"] for g in hard["gates"]]
        for key in ("pool", "fo", "ch", "cp", "cm", "cl"):
            for suffix, ls in (("", STYLE[key][2]), ("_t1", ":")):
                if suffix and key not in ("pool", "cm", "cl"):
                    continue
                d = load(DATA, "init", f"init_{kj}_{key}{suffix}.json")
                if d is None:
                    continue
                lab, color, _ = STYLE[key]
                y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
                ax.plot(range(len(y)), y, ls, color=color,
                        lw=1.5 if key in ("pool", "cl", "cm") and not suffix else 1.0,
                        label=(lab if not suffix else None) if kj == "k32_j224" else None)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(8))
        ax.set_xlabel(r"gate $\ell$ (0 = input side)")
        ax.set_title(title)
    axes[0].set_ylabel(r"RMS $\partial\mathcal{L}/\partial c_{\ell+1}$ / hard")
    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.0))
    labels.append(r"same, $T{=}1$")
    axes[0].legend(handles, labels, loc="upper right", fontsize=4.9, ncol=1)
    save(fig, "fig_init")


def table_init():
    """Per-block growth factor and surrogate energy, for the note."""
    for kj in ("k32_j224", "k256_j256"):
        print(kj)
        for key in ("hard", "pool", "fo", "cl", "ch", "cp", "cm"):
            for suffix in ("", "_t1"):
                d = load(DATA, "init", f"init_{kj}_{key}{suffix}.json")
                if d is None:
                    continue
                hard = load(DATA, "init", f"init_{kj}_hard.json")
                g0 = d["gates"][0]["code_grad_rms"] / hard["gates"][0]["code_grad_rms"]
                sig = sum(g["sigma"] for g in d["gates"]) / len(d["gates"])
                print(f"  {key + suffix:10s} gate0/hard {g0:7.2f}  per block x{d.get('per_block_factor', float('nan')):.3f}"
                      f"  sigma {sig:.3f}")


# --------------------------------------------------------------------------- #
# Figure 2: the screens, against the hard baselines of the code-carried grid
# --------------------------------------------------------------------------- #
CELLS = {(32, 224): ("ma_cr_hard_k32_pnorm", "ma_cr_hard_k256_pnorm"),
         (64, 64): ("ma_cr_hard_k64_pnorm", "ma_cr_hard_k128_pnorm"),
         (64, 448): ("ma_cr_hard_k64_pnorm", "ma_cr_hard_k512_pnorm"),
         (128, 128): ("ma_cr_hard_k128_pnorm", "ma_cr_hard_k256_pnorm"),
         (128, 384): ("ma_cr_hard_k128_pnorm", "ma_cr_hard_k512_pnorm"),
         (256, 256): ("ma_cr_hard_k256_pnorm", "ma_cr_hard_k512_pnorm")}


def curves():
    base = load(KJ, "curves_new.json") or {}
    new = load(DATA, "curves_bergen.json") or {}
    return {**base, **new}


def run_of(cell, key):
    K, J = cell
    if key == "pool":
        return f"ma_cr_rbk_k{K}_j{J}_t2"
    if key == "fo":
        return f"ma_cr_rbk_k{K}_j{J}_t2_fo"
    return f"be_cr_rbk_k{K}_j{J}_t2_{key}"


def series(cv, run):
    c = cv.get(run)
    return {} if c is None else dict((int(s), v) for s, v in c["val"])


def delta_panel(ax, cv, cell, keys, max_step=3000, show_pool_ref=True):
    K, J = cell
    hard_k, hard_kj = CELLS[cell]
    h = series(cv, hard_k)
    for key in keys:
        d = series(cv, run_of(cell, key))
        pts = [(s, v - h[s]) for s, v in sorted(d.items()) if s in h and s <= max_step]
        if not pts:
            continue
        lab, color, ls = STYLE[key]
        ax.plot(*zip(*pts), ls, color=color, lw=1.5 if key in ("pool", "cl", "cm") else 1.0,
                marker="o", ms=2.2, mfc="white", mew=0.7, label=lab)
    if show_pool_ref:
        hk = series(cv, hard_kj)
        pts = [(s, v - h[s]) for s, v in sorted(hk.items()) if s in h and s <= max_step]
        if pts:
            ax.plot(*zip(*pts), "--", color=INK, lw=0.9, label=rf"hard Top-{K + J}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(rf"val CE $-$ hard Top-{K}")


def fig_screens(max_step=3000):
    cv = curves()
    Ks = [cell for cell in ((32, 224), (64, 64), (128, 128), (256, 256))
          if any(run_of(cell, k) in cv for k in ("cl", "cm"))]
    if not Ks:
        return
    fig, axes = plt.subplots(1, len(Ks), figsize=(1.9 * len(Ks) + 0.2, 1.7),
                             gridspec_kw={"wspace": 0.5}, squeeze=False)
    keys = ("pool", "fo", "ch", "cp", "cm05", "cm", "cl")
    for ax, cell, letter in zip(axes[0], Ks, "abcd"):
        delta_panel(ax, cv, cell, keys, max_step, show_pool_ref=True)
        ax.set_title(rf"{letter}  $K{{=}}{cell[0]}$, $J{{=}}{cell[1]}$")
        ax.set_xlim(400, max_step + 50)
    # one legend, in the last panel, with proxies for every scope that appears anywhere
    from matplotlib.lines import Line2D
    present = [k for k in keys if any(run_of(cell, k) in cv for cell in Ks)]
    handles = [Line2D([], [], color=STYLE[k][1], ls=STYLE[k][2], marker="o", ms=2.2, mfc="white",
                      mew=0.7, lw=1.3, label=STYLE[k][0]) for k in present]
    handles.append(Line2D([], [], color=INK, ls="--", lw=0.9, label=r"hard Top-$(K{+}J)$"))
    axes[0][-1].legend(handles=handles, loc="upper right", fontsize=5.0, ncol=1)
    save(fig, "fig_screens")


def table_screens(steps=(1000, 2000, 3000)):
    cv = curves()
    for cell in sorted(CELLS):
        K, J = cell
        h = series(cv, CELLS[cell][0])
        hk = series(cv, CELLS[cell][1])
        rows = []
        for key in ("pool", "fo", "cl", "cm", "cm05", "cp", "ch"):
            d = series(cv, run_of(cell, key))
            if not d:
                continue
            rows.append((key, [round(d[s] - h[s], 4) if s in d and s in h else None for s in steps],
                         [round(d[s], 4) if s in d else None for s in steps]))
        if rows:
            print(f"K={K} J={J}  hard Top-K at {steps}: {[round(h.get(s, float('nan')), 4) for s in steps]}"
                  f"  hard Top-(K+J): {[round(hk.get(s, float('nan')) - h.get(s, float('nan')), 4) for s in steps]}")
            for key, dv, v in rows:
                print(f"   {key:5s} delta {dv}  abs {v}")


# --------------------------------------------------------------------------- #
# Figure 3: runs of the carried code (analysis/code_runs.py)
# --------------------------------------------------------------------------- #
def fig_runs():
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.75), gridspec_kw={"wspace": 0.4})
    rows = [  # file stem, label, color, ls
        ("ma_cr_hard_k32_pnorm_step20000", r"hard Top-32, 20k", INK, "-"),
        ("ma_cr_hard_k32_pnorm_step4000", r"hard Top-32, 4k", INK, ":"),
        ("ma_cr_hard_k256_pnorm_step20000", r"hard Top-256, 20k", MUTED, "-"),
        ("ma_cr_hard_k256_pnorm_step4000", r"hard Top-256, 4k", MUTED, ":"),
        ("ma_cr_rbk_k32_j224_t2_step20000", r"pool (32, 224), 20k", ORANGE, "-"),
        ("ma_cr_rbk_k32_j224_t2_step4000", r"pool (32, 224), 4k", ORANGE, ":"),
        ("ma_cr_rbk_k256_j256_t2_step4000", r"pool (256, 256), 4k", "#f2a37e", ":"),
        ("ma_cr_rbk_k32_j224_t2_fo_step4000", r"first order (32, 224), 4k", YELLOW, ":"),
    ]
    # the carry scopes at step 3000, whichever have been measured
    import glob
    for f in sorted(glob.glob(os.path.join(DATA, "runs", "be_cr_rbk_*_step3000.json"))):
        stem = os.path.basename(f)[:-5]
        parts = stem.split("_")            # be cr rbk kK jJ t2 <suffix> stepN
        K, J, suf = parts[3][1:], parts[4][1:], parts[6]
        if suf not in STYLE:
            continue
        lab, color, _ = STYLE[suf]
        rows.append((stem, rf"{lab} ({K}, {J}), 3k", color, "-" if K == "32" else "--"))
    for key, panel in (("survival", 0), ("evict_frac", 1)):
        ax = axes[panel]
        for stem, lab, color, ls in rows:
            d = load(DATA, "runs", stem + ".json")
            if d is None:
                continue
            y = d[key]
            x = list(range(len(y)))
            if key == "evict_frac":
                x, y = x[1:], y[1:]
            ax.plot(x, y, ls, color=color, lw=1.2, marker="o", ms=2.0, mfc="white", mew=0.6,
                    label=lab if panel == 0 else None)
    axes[0].set_xlabel(r"gates after entry $d$")
    axes[0].set_ylabel(r"P(still in the support)")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("a  survival of an entered feature")
    axes[0].legend(fontsize=4.3, loc="upper right", ncol=1)
    axes[1].set_xlabel(r"gate $\ell$")
    axes[1].set_ylabel(r"share of gate $\ell{-}1$'s support evicted")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title("b  turnover per gate")
    save(fig, "fig_runs")


def table_runs():
    import glob
    for f in sorted(glob.glob(os.path.join(DATA, "runs", "*.json"))):
        d = json.load(open(f))
        surv = d["survival"]
        exp_reads = 1 + sum(surv[1:])
        print(f"  {os.path.basename(f)[:-5]:40s} survival d=1 {surv[1]:.2f}  expected downstream reads of an entry at gate 0 "
              f"{exp_reads:.2f}  evict/gate {sum(d['evict_frac'][1:]) / (len(d['evict_frac']) - 1):.2f}  "
              f"final from gate0 {d['final_entry'][0]:.2f} last gate {d['final_entry'][-1]:.2f}  reentry {d['reentry_frac']:.2f}")


def latex_table(step=3000):
    """Rows of the note's table: val CE at `step` minus hard Top-K, per cell and scope."""
    cv = curves()
    cols = ("pool", "fo", "cl", "cm", "cm05", "cp", "ch")
    print("% K & J & " + " & ".join(cols) + " & hard Top-(K+J) \\\\")
    for cell in sorted(CELLS):
        K, J = cell
        hard_k, hard_kj = CELLS[cell]
        h = series(cv, hard_k)
        if step not in h:
            continue
        cells = []
        for key in cols:
            d = series(cv, run_of(cell, key))
            cells.append(f"{d[step] - h[step]:+.3f}" if step in d else "--")
        hk = series(cv, hard_kj)
        cells.append(f"{hk[step] - h[step]:+.3f}" if step in hk else "--")
        print(f"{K} & {J} & " + " & ".join(cells) + f" \\\\   % hard Top-{K} = {h[step]:.4f}")


# --------------------------------------------------------------------------- #
# Figure 4: the 10k continuations against the grid's 20k runs
# --------------------------------------------------------------------------- #
def fig_long(max_step=10000):
    cv = curves()
    cells = [c for c in ((32, 224), (256, 256)) if f"be_cr_rbk_k{c[0]}_j{c[1]}_t2_cm_long" in cv]
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells), figsize=(2.5 * len(cells) + 0.2, 1.7),
                             gridspec_kw={"wspace": 0.5}, squeeze=False)
    for ax, cell, letter in zip(axes[0], cells, "ab"):
        K, J = cell
        h = series(cv, CELLS[cell][0])
        refs = {32: [(64, "#f0a0a0"), (128, "#d05050"), (256, "#800000")],
                256: [(512, "#800000")]}[K]
        for Kp, color in refs:
            r = series(cv, f"ma_cr_hard_k{Kp}_pnorm")
            pts = [(s_, v - h[s_]) for s_, v in sorted(r.items()) if s_ in h and s_ <= max_step]
            ax.plot(*zip(*pts), "--", color=color, lw=0.9, label=rf"hard Top-{Kp}")
        for key, run in (("pool", run_of(cell, "pool")), ("cm", f"be_cr_rbk_k{K}_j{J}_t2_cm_long")):
            d = series(cv, run)
            pts = [(s_, v - h[s_]) for s_, v in sorted(d.items()) if s_ in h and s_ <= max_step]
            if pts:
                lab, color, ls = STYLE[key]
                ax.plot(*zip(*pts), ls, color=color, lw=1.4, marker="o", ms=2.0, mfc="white", mew=0.6,
                        label=lab + (" (10k run)" if key == "cm" else ""))
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks([0, 2500, 5000, 7500, 10000])
        ax.set_xticklabels(["0", "2.5k", "5k", "7.5k", "10k"])
        ax.set_xlabel("step")
        ax.set_ylabel(rf"val CE $-$ hard Top-{K}")
        ax.set_title(rf"{letter}  $K{{=}}{K}$, $J{{=}}{J}$")
        ax.legend(fontsize=4.8, loc="best")
    save(fig, "fig_long")


def fig_training():
    cv = curves()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.45), gridspec_kw={"wspace": 0.55})
    keys = ("pool", "fo", "ch", "cp", "cm05", "cm", "cl")
    for ax, cell, letter in zip(axes[:2], ((32, 224), (256, 256)), "ab"):
        delta_panel(ax, cv, cell, keys, 3000, show_pool_ref=True)
        ax.set_title(rf"{letter}  $K{{=}}{cell[0]}$, $J{{=}}{cell[1]}$, to 3k")
        ax.set_xlim(400, 3050)
        ax.set_xticks([1000, 2000, 3000])
        ax.set_xticklabels(["1k", "2k", "3k"])
    from matplotlib.lines import Line2D
    present = [k for k in keys if any(run_of(c, k) in cv for c in ((32, 224), (256, 256)))]
    handles = [Line2D([], [], color=STYLE[k][1], ls=STYLE[k][2], marker="o", ms=2.2, mfc="white",
                      mew=0.7, lw=1.3, label=STYLE[k][0]) for k in present]
    handles.append(Line2D([], [], color=INK, ls="--", lw=0.9, label=r"hard Top-$(K{+}J)$"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=4.6, ncol=1)
    for ax, cell, letter in zip(axes[2:], ((32, 224), (256, 256)), "cd"):
        K, J = cell
        h = series(cv, CELLS[cell][0])
        refs = {32: [(64, "#f0a0a0"), (128, "#d05050")], 256: [(512, "#800000")]}[K]
        for Kp, color in refs:
            r = series(cv, f"ma_cr_hard_k{Kp}_pnorm")
            pts = [(s_, v - h[s_]) for s_, v in sorted(r.items()) if s_ in h and s_ <= 10000]
            ax.plot(*zip(*pts), "--", color=color, lw=0.9, label=rf"hard Top-{Kp}")
        for key, run in (("pool", run_of(cell, "pool")), ("cm", f"be_cr_rbk_k{K}_j{J}_t2_cm_long")):
            d = series(cv, run)
            pts = [(s_, v - h[s_]) for s_, v in sorted(d.items()) if s_ in h and s_ <= 10000]
            if pts:
                lab, color, ls = STYLE[key]
                ax.plot(*zip(*pts), ls, color=color, lw=1.4, marker="o", ms=1.8, mfc="white", mew=0.6,
                        label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks([0, 5000, 10000])
        ax.set_xticklabels(["0", "5k", "10k"])
        ax.set_xlabel("step")
        ax.set_ylabel(rf"val CE $-$ hard Top-{K}")
        ax.set_title(rf"{letter}  $K{{=}}{K}$, $J{{=}}{J}$, to 10k")
        ax.legend(fontsize=4.4, loc="best")
    save(fig, "fig_training")


def fig_mech():
    """One row: the backward at initialization (a, b) and the runs of the code (c, d)."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.42), gridspec_kw={"wspace": 0.5})
    # (a, b) initialization
    for ax, (kj, title) in zip(axes[:2], [("k32_j224", r"a  step 0, $K{=}32$, $J{=}224$"),
                                          ("k256_j256", r"b  step 0, $K{=}256$, $J{=}256$")]):
        hard = load(DATA, "init", f"init_{kj}_hard.json")
        h = [g["code_grad_rms"] for g in hard["gates"]]
        for key in ("pool", "fo", "ch", "cp", "cm", "cl"):
            for suffix, ls in (("", STYLE[key][2]), ("_t1", ":")):
                if suffix and key not in ("pool", "cm", "cl"):
                    continue
                d = load(DATA, "init", f"init_{kj}_{key}{suffix}.json")
                if d is None:
                    continue
                lab, color, _ = STYLE[key]
                y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
                ax.plot(range(len(y)), y, ls, color=color,
                        lw=1.4 if key in ("pool", "cl", "cm") and not suffix else 0.9,
                        label=(lab if not suffix else None) if kj == "k32_j224" else None)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(0, 8, 1))
        ax.set_xlabel(r"gate $\ell$ (0 = input side)")
        ax.set_title(title)
    axes[0].set_ylabel(r"RMS $\partial\mathcal{L}/\partial c_{\ell+1}$ / hard")
    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.0)); labels.append(r"same, $T{=}1$")
    axes[0].legend(handles, labels, loc="upper right", fontsize=4.3, ncol=1, handlelength=1.3)
    # (c, d) runs
    rows = [
        ("ma_cr_hard_k32_pnorm_step20000", r"hard Top-32, 20k", INK, "-"),
        ("ma_cr_hard_k256_pnorm_step20000", r"hard Top-256, 20k", MUTED, "-"),
        ("ma_cr_rbk_k32_j224_t2_step20000", r"pool (32, 224), 20k", ORANGE, "-"),
        ("ma_cr_rbk_k256_j256_t2_step4000", r"pool (256, 256), 4k", "#f2a37e", ":"),
        ("ma_cr_rbk_k32_j224_t2_fo_step4000", r"first order (32, 224), 4k", YELLOW, ":"),
        ("be_cr_rbk_k32_j224_t2_cl_step3000", r"carry, local (32, 224), 3k", BLUE, "-"),
        ("be_cr_rbk_k32_j224_t2_cm_step3000", r"carry, mixed (32, 224), 3k", AQUA, "-"),
        ("be_cr_rbk_k256_j256_t2_cl_step3000", r"carry, local (256, 256), 3k", BLUE, "--"),
        ("be_cr_rbk_k256_j256_t2_cm_step3000", r"carry, mixed (256, 256), 3k", AQUA, "--"),
    ]
    for key, ax in (("survival", axes[2]), ("evict_frac", axes[3])):
        for stem, lab, color, ls in rows:
            d = load(DATA, "runs", stem + ".json")
            if d is None:
                continue
            y = d[key]; x = list(range(len(y)))
            if key == "evict_frac":
                x, y = x[1:], y[1:]
            ax.plot(x, y, ls, color=color, lw=1.1, marker="o", ms=1.8, mfc="white", mew=0.6,
                    label=lab if key == "survival" else None)
    axes[2].set_xlabel(r"gates after entry $d$")
    axes[2].set_ylabel("P(still in the support)")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("c  survival of an entry")
    axes[2].legend(fontsize=3.9, loc="upper right", handlelength=1.3, labelspacing=0.25)
    axes[3].set_xlabel(r"gate $\ell$")
    axes[3].set_ylabel(r"share of support evicted")
    axes[3].set_ylim(0, 0.85)
    axes[3].set_xticks(range(1, 8))
    axes[3].set_title("d  turnover per gate")
    save(fig, "fig_mech")


# --------------------------------------------------------------------------- #
# Figure 5: alternatives to the surrogate (stochastic width, soft STE, candidates-only)
# --------------------------------------------------------------------------- #
ALT_KEYS = ("sw_uni", "sw_two", "sw_two25", "sw_geo", "sw_geo16", "ste", "inact")


def alt_run_of(cell, key):
    K, J = cell
    if key.startswith("sw_"):
        return f"ma_alt_{key}_k{K}_j{J}"
    if key == "ste":
        return f"ma_alt_ste_k{K}_j{J}_t2"
    if key == "inact":
        return f"ma_alt_inact_k{K}_j{J}_t2"
    return run_of(cell, key)


def fig_alt(max_step=3000):
    cv = {**curves(), **(load(DATA, "curves_alt.json") or {})}
    cells = [c for c in ((32, 224), (128, 128), (256, 256)) if any(alt_run_of(c, k) in cv for k in ALT_KEYS)]
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells), figsize=(2.4 * len(cells) + 0.3, 1.85),
                             gridspec_kw={"wspace": 0.5}, squeeze=False)
    for ax, cell, letter in zip(axes[0], cells, "abc"):
        K, J = cell
        hard_k, hard_kj = CELLS[cell]
        h = series(cv, hard_k)
        for key in ("pool", "cm") + ALT_KEYS:
            run = alt_run_of(cell, key) if key in ALT_KEYS else run_of(cell, key)
            if key == "cm":
                run = f"be_cr_rbk_k{K}_j{J}_t2_cm"
            d = series(cv, run)
            pts = [(s_, v - h[s_]) for s_, v in sorted(d.items()) if s_ in h and s_ <= max_step]
            if not pts:
                continue
            lab, color, ls = STYLE[key]
            ax.plot(*zip(*pts), ls, color=color, lw=1.5 if key in ("pool", "cm") else 1.2,
                    marker="o", ms=2.0, mfc="white", mew=0.6, label=lab)
        hk = series(cv, hard_kj)
        pts = [(s_, v - h[s_]) for s_, v in sorted(hk.items()) if s_ in h and s_ <= max_step]
        if pts:
            ax.plot(*zip(*pts), "--", color=INK, lw=0.9, label=rf"hard Top-{K + J}")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlim(400, max_step + 50)
        ax.set_xticks([1000, 2000, 3000]); ax.set_xticklabels(["1k", "2k", "3k"])
        ax.set_xlabel("step")
        ax.set_ylabel(rf"val CE $-$ hard Top-{K}")
        ax.set_title(rf"{letter}  $K{{=}}{K}$, $J{{=}}{J}$")
    from matplotlib.lines import Line2D
    present = [k for k in ("pool", "cm") + ALT_KEYS if any((alt_run_of(c, k) if k in ALT_KEYS else (f"be_cr_rbk_k{c[0]}_j{c[1]}_t2_cm" if k == "cm" else run_of(c, k))) in cv for c in cells)]
    handles = [Line2D([], [], color=STYLE[k][1], ls=STYLE[k][2], marker="o", ms=2.0, mfc="white", mew=0.6, lw=1.2, label=STYLE[k][0]) for k in present]
    handles.append(Line2D([], [], color=INK, ls="--", lw=0.9, label=r"hard Top-$(K{+}J)$"))
    axes[0][-1].legend(handles=handles, fontsize=4.2, loc="upper right", ncol=1)
    save(fig, "fig_alt")


def table_alt(step=3000):
    cv = {**curves(), **(load(DATA, "curves_alt.json") or {})}
    for cell in ((32, 224), (64, 64), (128, 128), (256, 256)):
        K, J = cell
        h = series(cv, CELLS[cell][0]); hk = series(cv, CELLS[cell][1])
        if step not in h:
            continue
        row = []
        for key in ("pool", "fo", "cm") + ALT_KEYS:
            run = f"be_cr_rbk_k{K}_j{J}_t2_cm" if key == "cm" else (alt_run_of(cell, key) if key in ALT_KEYS else run_of(cell, key))
            d = series(cv, run)
            row.append(f"{key} {d[step] - h[step]:+.3f}" if step in d else f"{key} --")
        print(f"K={K} J={J}: " + "  ".join(row) + f"  hard{K + J} {hk[step] - h[step]:+.3f}")


def fig_widths():
    """Validation CE of one checkpoint at several inference widths, against the hard ladder."""
    import glob
    cv = curves()
    files = sorted(glob.glob(os.path.join(DATA, "alt", "widths_ma_alt_sw_*.json")))
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.8), gridspec_kw={"wspace": 0.45})
    ladder = {Kp: series(cv, f"ma_cr_hard_k{Kp}_pnorm").get(3000) for Kp in (32, 64, 128, 256, 512)}
    for ax, K in zip(axes, (32, 256)):
        xs = [Kp for Kp in sorted(ladder) if Kp >= K and ladder[Kp] is not None]
        ax.plot(xs, [ladder[Kp] for Kp in xs], "--o", color=INK, ms=2.5, lw=0.9, label="hard Top-$K'$ trained at $K'$")
        for f in files:
            d = json.load(open(f))
            if d["k_train"] != K:
                continue
            key = "sw_two" if "_two_" in f else ("sw_uni" if "_uni_" in f else "sw_geo")
            w = sorted((int(a), b) for a, b in d["widths"].items())
            lab, color, _ = STYLE[key]
            ax.plot([a for a, _ in w], [b for _, b in w], "-o", color=color, ms=2.5, lw=1.2, label=lab + f", trained at $K{{=}}{K}$")
        ax.set_xscale("log", base=2)
        ticks = xs if K == 32 else [256, 320, 384, 448, 512]
        ax.set_xticks(ticks); ax.set_xticklabels([str(x) for x in ticks]); ax.minorticks_off()
        ax.set_xlabel(r"inference width $K'$")
        ax.set_ylabel("val CE at step 3000")
        ax.set_title(rf"{'ab'[K == 256]}  models trained at $K{{=}}{K}$")
        ax.legend(fontsize=4.8, loc="best")
    save(fig, "fig_widths")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_init()
    table_init()
    fig_screens()
    table_screens()
    fig_runs()
    table_runs()
    latex_table()
    fig_long()
    fig_training()
    fig_mech()
    fig_alt()
    table_alt()
    fig_widths()
