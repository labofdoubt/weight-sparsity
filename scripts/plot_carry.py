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
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw={"wspace": 0.42})
    for ax, (kj, title) in zip(axes[:2], [("k32_j224", r"a  $K{=}32$, $J{=}224$, $T{=}2$"),
                                          ("k256_j256", r"b  $K{=}256$, $J{=}256$, $T{=}2$")]):
        hard = load(DATA, "init", f"init_{kj}_hard.json")
        h = [g["code_grad_rms"] for g in hard["gates"]]
        for key in ("pool", "fo", "ch", "cp", "cm", "cl"):
            d = load(DATA, "init", f"init_{kj}_{key}.json")
            if d is None:
                continue
            lab, color, ls = STYLE[key]
            y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
            ax.plot(range(len(y)), y, ls, color=color, lw=1.5 if key in ("pool", "cl", "cm") else 1.0,
                    label=lab)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(8))
        ax.set_xlabel(r"gate $\ell$ (0 = input side, 7 = last)")
        ax.set_ylabel(r"RMS $\partial\mathcal{L}/\partial c_{\ell+1}$ / hard")
        ax.set_title(title)
        if kj == "k32_j224":
            ax.legend(loc="upper right", fontsize=5.4)
    # (c) the kernel width: T = 1 versus T = 2 at K = 32
    ax = axes[2]
    hard = load(DATA, "init", "init_k32_j224_hard.json")
    h = [g["code_grad_rms"] for g in hard["gates"]]
    for key, suffix, ls in (("pool", "", "-"), ("pool", "_t1", ":"), ("cm", "", "-"), ("cm", "_t1", ":"),
                            ("cl", "", "-"), ("cl", "_t1", ":")):
        d = load(DATA, "init", f"init_k32_j224_{key}{suffix}.json")
        if d is None:
            continue
        lab, color, _ = STYLE[key]
        y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
        ax.plot(range(len(y)), y, ls, color=color, lw=1.3,
                label=lab + (r", $T{=}1$" if suffix else r", $T{=}2$"))
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(8))
    ax.set_xlabel(r"gate $\ell$")
    ax.set_title(r"c  $K{=}32$: kernel width")
    ax.legend(loc="upper right", fontsize=5.2)
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
CELLS = {32: (224, "ma_cr_hard_k32_pnorm", "ma_cr_hard_k256_pnorm"),
         64: (448, "ma_cr_hard_k64_pnorm", "ma_cr_hard_k512_pnorm"),
         128: (384, "ma_cr_hard_k128_pnorm", "ma_cr_hard_k512_pnorm"),
         256: (256, "ma_cr_hard_k256_pnorm", "ma_cr_hard_k512_pnorm")}


def curves():
    base = load(KJ, "curves_new.json") or {}
    new = load(DATA, "curves_bergen.json") or {}
    return {**base, **new}


def run_of(K, key):
    J = CELLS[K][0]
    if key == "pool":
        return f"ma_cr_rbk_k{K}_j{J}_t2"
    if key == "fo":
        return f"ma_cr_rbk_k{K}_j{J}_t2_fo"
    return f"be_cr_rbk_k{K}_j{J}_t2_{key}"


def series(cv, run):
    c = cv.get(run)
    return {} if c is None else dict((int(s), v) for s, v in c["val"])


def delta_panel(ax, cv, K, keys, max_step=3000, show_pool_ref=True):
    J, hard_k, hard_kj = CELLS[K]
    h = series(cv, hard_k)
    for key in keys:
        d = series(cv, run_of(K, key))
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
    Ks = [K for K in (32, 128, 256) if any(run_of(K, k) in cv for k in ("cl", "cm"))]
    if not Ks:
        return
    fig, axes = plt.subplots(1, len(Ks), figsize=(2.4 * len(Ks) + 0.2, 2.0),
                             gridspec_kw={"wspace": 0.45}, squeeze=False)
    for ax, K, letter in zip(axes[0], Ks, "abc"):
        delta_panel(ax, cv, K, ("pool", "fo", "ch", "cp", "cm", "cl"), max_step)
        ax.set_title(rf"{letter}  $K{{=}}{K}$, $J{{=}}{CELLS[K][0]}$")
        ax.set_xlim(400, max_step + 50)
        if K == Ks[0]:
            ax.legend(loc="lower left", fontsize=5.0, ncol=2)
    save(fig, "fig_screens")


def table_screens(steps=(1000, 2000, 3000)):
    cv = curves()
    for K in (32, 64, 128, 256):
        h = series(cv, CELLS[K][1])
        hk = series(cv, CELLS[K][2])
        rows = []
        for key in ("pool", "fo", "cl", "cm", "cp", "ch"):
            d = series(cv, run_of(K, key))
            if not d:
                continue
            rows.append((key, [round(d[s] - h[s], 4) if s in d and s in h else None for s in steps],
                         [round(d[s], 4) if s in d else None for s in steps]))
        if rows:
            print(f"K={K}  hard Top-K at {steps}: {[round(h.get(s, float('nan')), 4) for s in steps]}"
                  f"  hard Top-(K+J): {[round(hk.get(s, float('nan')) - h.get(s, float('nan')), 4) for s in steps]}")
            for key, dv, v in rows:
                print(f"   {key:5s} delta {dv}  abs {v}")


# --------------------------------------------------------------------------- #
# Figure 3: runs of the carried code (analysis/code_runs.py)
# --------------------------------------------------------------------------- #
def fig_runs():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.95), gridspec_kw={"wspace": 0.45})
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
    for key, panel in (("survival", 0), ("evict_frac", 1), ("final_entry", 2)):
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
    axes[0].legend(fontsize=4.6, loc="upper right", ncol=1)
    axes[1].set_xlabel(r"gate $\ell$")
    axes[1].set_ylabel(r"share of gate $\ell{-}1$'s support evicted")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title("b  turnover per gate")
    axes[2].set_xlabel(r"gate of entry")
    axes[2].set_ylabel("share of the final code")
    axes[2].set_title("c  final code by gate of entry")
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
    for K in (32, 64, 128, 256):
        J, hard_k, hard_kj = CELLS[K]
        h = series(cv, hard_k)
        if step not in h:
            continue
        cells = []
        for key in cols:
            d = series(cv, run_of(K, key))
            cells.append(f"{d[step] - h[step]:+.3f}" if step in d else "--")
        hk = series(cv, hard_kj)
        cells.append(f"{hk[step] - h[step]:+.3f}" if step in hk else "--")
        print(f"{K} & {J} & " + " & ".join(cells) + f" \\\\   % hard Top-{K} = {h[step]:.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_init()
    table_init()
    fig_screens()
    table_screens()
    fig_runs()
    table_runs()
    latex_table()
