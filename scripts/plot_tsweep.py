"""Figures of the (reach, strength) grid for code-carried RBLapSum, 2026-10-05.

Reads docs/figures/tsweep/data/curves_tsweep.json (the 48 ``ma_ts_*`` runs,
written by scripts/extract_val_curves.py) and the hard Top-K' ladder of the
code-carried K+J grid (docs/figures/kj-cr/data/curves_new.json); writes
docs/figures/tsweep/fig_*.{pdf,png} and prints the step-3000 table.

Runs: ma_ts_{span|b}_t{025|050|100}_s{010|025|050|100}_k{32|256}_j{480|256},
10k steps of the 20k schedule (train_guard --stop-step 10000), validation every 500.
The kernel width is T = tau * span (span = s_(K+1) - s_(K+J)) or T = tau * b
(b = s_(K+1)); the support strength s sets gamma = 2 s T / b per token, so the
(K+1)-th member's support gradient equals s times its read gradient.

  python scripts/plot_tsweep.py            # all figures and the table
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "tsweep", "data")
KJ = os.path.join(ROOT, "docs", "figures", "kj-cr", "data")
OUT = os.path.join(ROOT, "docs", "figures", "tsweep")

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

TAUS = [("025", 0.25), ("050", 0.5), ("100", 1.0)]
SS = [("010", 0.1), ("025", 0.25), ("050", 0.5), ("100", 1.0)]
CELLS = {32: 480, 256: 256}
TAU_COLOR = {0.25: BLUE, 0.5: ORANGE, 1.0: AQUA}
MODE_LABEL = {"span": r"$T=\tau\,(s_{(K+1)}-s_{(K+J)})$", "b": r"$T=\tau\, s_{(K+1)}$"}
HARD_KS = [32, 64, 128, 256, 512]


def load(*parts):
    p = os.path.join(*parts)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def curves():
    return {**(load(KJ, "curves_new.json") or {}), **(load(DATA, "curves_tsweep.json") or {})}


def run_of(mode, tau, s, K):
    t = dict(TAUS)
    tk = [k for k, v in TAUS if v == tau][0]
    sk = [k for k, v in SS if v == s][0]
    return f"ma_ts_{mode}_t{tk}_s{sk}_k{K}_j{CELLS[K]}"


def hard_of(K):
    return f"ma_cr_hard_k{K}_pnorm"


def pool_of(K):
    return f"ma_cr_rbk_k{K}_j{CELLS[K]}_t2"


def series(cv, run):
    c = cv.get(run)
    return {} if c is None else dict((int(st), v) for st, v in c["val"])


def shade(color, frac):
    """Blend `color` towards white: frac=1 is the full color, frac=0 is white."""
    r, g, b = to_rgb(color)
    return (1 - frac * (1 - r), 1 - frac * (1 - g), 1 - frac * (1 - b))


def s_shade(s):
    return {0.1: 0.35, 0.25: 0.55, 0.5: 0.78, 1.0: 1.0}[s]


def grid_panel(ax, cv, mode, K, max_step=10000, ref="hard"):
    """12 curves: val CE minus the reference, colour = tau, shade = s."""
    h = series(cv, hard_of(K) if ref == "hard" else ref)
    for tk, tau in TAUS:
        for sk, s in SS:
            d = series(cv, run_of(mode, tau, s, K))
            pts = [(st, v - h[st]) for st, v in sorted(d.items()) if st in h and st <= max_step]
            if not pts:
                continue
            ax.plot(*zip(*pts), "-", color=shade(TAU_COLOR[tau], s_shade(s)),
                    lw=1.0 + 0.5 * s, marker="o", ms=1.8 + 1.2 * s, mfc="white", mew=0.6,
                    label=rf"$\tau={tau:g}$, $s={s:g}$")
    p = series(cv, pool_of(K))
    pts = [(st, v - h[st]) for st, v in sorted(p.items()) if st in h and st <= max_step]
    if pts:
        ax.plot(*zip(*pts), ":", color=INK2, lw=1.0, label=rf"pool, $T=2$, $\gamma=1$ (reference)")
    # hard Top-(K+J) is left out on purpose: at K=32 it sits 0.17 below and
    # would compress the grid into a strip; it appears in fig_ts_best.
    ax.axhline(0, color=INK, lw=0.8, label=rf"hard Top-{K}")
    ax.set_xlabel("step")
    ax.set_ylabel(rf"val CE $-$ hard Top-{K}")
    ax.set_xlim(0, max_step)


def fig_grid(mode, K, max_step=10000):
    cv = curves()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    grid_panel(ax, cv, mode, K, max_step)
    ax.set_title(rf"$K={K}$, $J={CELLS[K]}$, {MODE_LABEL[mode]}, $\gamma=2sT/b$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.0)
    save(fig, f"fig_ts_{mode}_k{K}")


def table(step=10000):
    cv = curves()
    rows = []
    for mode in ("span", "b"):
        for K in CELLS:
            h = series(cv, hard_of(K)).get(step)
            for tk, tau in TAUS:
                for sk, s in SS:
                    r = run_of(mode, tau, s, K)
                    d = series(cv, r)
                    c = cv.get(r)
                    v = d.get(step)
                    dg = c["diag"] if c else []
                    last = dg[-1] if dg else [None] * 5
                    rows.append((mode, K, tau, s, v, None if (v is None or h is None) else v - h,
                                 last[1], last[2], last[3], c.get("diverged") if c else None))
    return rows


def best_cells(step=10000):
    cv = curves()
    best = {}
    for mode in ("span", "b"):
        for K in CELLS:
            cand = []
            for tk, tau in TAUS:
                for sk, s in SS:
                    v = series(cv, run_of(mode, tau, s, K)).get(step)
                    if v is not None:
                        cand.append((v, tau, s))
            if cand:
                best[(mode, K)] = min(cand)
    return best


def shared_legend(fig, axes):
    """The hard ladder is common to both panels; the best cells differ per panel."""
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    hard = [(h, l) for h, l in zip(h0, l0) if l.startswith("hard")]
    best0 = [(h, "span: " + l) for h, l in zip(h0, l0) if not l.startswith("hard")]
    best1 = [(h, "b: " + l) for h, l in zip(h1, l1) if not l.startswith("hard")]
    items = hard + best0 + best1
    fig.legend([h for h, _ in items], [l for _, l in items], loc="lower center",
               bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=6.0, frameon=False)


def fig_best(max_step=10000, step=10000):
    cv = curves()
    best = best_cells(step)
    ref = hard_of(512)
    h512 = series(cv, ref)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, mode in zip(axes, ("span", "b")):
        for i, K in enumerate(HARD_KS):
            d = series(cv, hard_of(K))
            pts = [(st, v - h512[st]) for st, v in sorted(d.items()) if st in h512 and st <= max_step]
            if pts:
                ax.plot(*zip(*pts), "-", color=shade(INK, 0.3 + 0.7 * (1 - i / (len(HARD_KS) - 1))),
                        lw=0.9, label=rf"hard Top-{K}")
        for K, color in ((32, MAGENTA), (256, YELLOW)):
            if (mode, K) not in best:
                continue
            v, tau, s = best[(mode, K)]
            d = series(cv, run_of(mode, tau, s, K))
            pts = [(st, v - h512[st]) for st, v in sorted(d.items()) if st in h512 and st <= max_step]
            ax.plot(*zip(*pts), "-", color=color, lw=1.6, marker="o", ms=2.4, mfc="white", mew=0.7,
                    label=rf"$K={K}$, $J={CELLS[K]}$: $\tau={tau:g}$, $s={s:g}$")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(MODE_LABEL[mode] + rf", best cells at step {step}")
        ax.set_xlabel("step")
        ax.set_xlim(0, max_step)
    axes[0].set_ylabel(r"val CE $-$ hard Top-512")
    shared_legend(fig, axes)
    save(fig, "fig_ts_best")
    # absolute version
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, mode in zip(axes, ("span", "b")):
        for i, K in enumerate(HARD_KS):
            d = series(cv, hard_of(K))
            pts = [(st, v) for st, v in sorted(d.items()) if st <= max_step and st >= 500]
            ax.plot(*zip(*pts), "-", color=shade(INK, 0.3 + 0.7 * (1 - i / (len(HARD_KS) - 1))),
                    lw=0.9, label=rf"hard Top-{K}")
        for K, color in ((32, MAGENTA), (256, YELLOW)):
            if (mode, K) not in best:
                continue
            v, tau, s = best[(mode, K)]
            d = series(cv, run_of(mode, tau, s, K))
            pts = [(st, v) for st, v in sorted(d.items()) if st <= max_step and st >= 500]
            ax.plot(*zip(*pts), "-", color=color, lw=1.6, marker="o", ms=2.4, mfc="white", mew=0.7,
                    label=rf"$K={K}$, $J={CELLS[K]}$: $\tau={tau:g}$, $s={s:g}$")
        ax.set_title(MODE_LABEL[mode] + rf", best cells at step {step}")
        ax.set_xlabel("step")
        ax.set_xlim(500, max_step)
    axes[0].set_ylabel("val CE")
    shared_legend(fig, axes)
    save(fig, "fig_ts_best_abs")


def main():
    os.makedirs(OUT, exist_ok=True)
    for mode in ("span", "b"):
        for K in CELLS:
            fig_grid(mode, K)
    fig_best()
    print(f"{'mode':5} {'K':>4} {'tau':>5} {'s':>5} {'val CE':>8} {'- hard':>8} {'T_eff':>7} {'gamma':>7} {'b':>7}  div")
    for mode, K, tau, s, v, dv, T, g, b, div in table():
        f = lambda x, w=8, p=4: f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f"{mode:5} {K:>4} {tau:>5g} {s:>5g} {f(v)} {f(dv)} {f(T, 7, 3)} {f(g, 7, 3)} {f(b, 7, 3)}  {'yes' if div else ''}")
    cv = curves()
    for (mode, K), (v, tau, s) in sorted(best_cells().items()):
        h = series(cv, hard_of(K)).get(10000)
        print(f"best {mode:5} K={K:>3}: tau={tau:g} s={s:g}  val {v:.4f}  ({v - h:+.4f} vs hard Top-{K})")


if __name__ == "__main__":
    main()
