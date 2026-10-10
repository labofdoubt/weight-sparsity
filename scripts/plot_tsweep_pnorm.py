"""The first-order span-rule cells with and without the post-norm, 2026-10-10.

Left column: the two first-order panels of docs/tsweep-reach-strength.tex
Section 7 (K = 32, J = 480 at tau = 0.5; K = 256, J = 256 at tau = 0.25), no
output norm, from docs/figures/tsweep/data/curves_tsweep.json.  Right column:
the same cells trained with post_norm = true (runs
``de_ts_span_t{050|025}_s{010,025,050,100}_k{K}_j{J}_fo_pnorm`` on denver, 10k
steps of the 20k schedule), from curves_tsweep_pnorm.json in the same folder.
Style as scripts/plot_tsweep.py: y = val CE minus the code-carried hard Top-K
(post-norm) at the same step; the four strengths as a blue ramp, two hard rungs
dashed red, the first-order T = 2, gamma = 1 run (no post-norm) dotted grey in
both columns as the common reference.

  python scripts/plot_tsweep_pnorm.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import plot_tsweep as pt  # noqa: E402

PANELS = [(32, 0.5), (256, 0.25)]          # (K, tau) per row
ARMS = [("no post-norm", "ma_ts", ""), ("post-norm", "de_ts", "_pnorm")]


def run_name(prefix, suffix, tau, s, K):
    tk = [k for k, v in pt.TAUS if v == tau][0]
    sk = [k for k, v in pt.SS if v == s][0]
    return f"{prefix}_span_t{tk}_s{sk}_k{K}_j{pt.CELLS[K]}_fo{suffix}"


def panel(ax, cv, K, tau, prefix, suffix, max_step=10000):
    scol = dict(zip([s for _, s in pt.SS], pt.ramp(pt.BLUES, len(pt.SS), *pt.B_RANGE)))
    hcol = dict(zip(pt.EXTRA_HARD[K], pt.ramp(pt.REDS, len(pt.EXTRA_HARD[K]), *pt.R_RANGE)))
    for kp in pt.EXTRA_HARD[K]:
        x, y = pt.delta(cv, pt.hard_of(kp), pt.hard_of(K), max_step)
        ax.plot(x, y, "--", color=hcol[kp], lw=1.6)
    x, y = pt.delta(cv, pt.ref_of(K, "fo"), pt.hard_of(K), max_step)
    ax.plot(x, y, ":", color=pt.GREY, lw=1.8)
    missing = []
    for _, s in pt.SS:
        run = run_name(prefix, suffix, tau, s, K)
        x, y = pt.delta(cv, run, pt.hard_of(K), max_step)
        if not x:
            missing.append(run)
        ax.plot(x, y, "-", color=scol[s], lw=2.0, marker="o", ms=3, mfc="white", mew=0.8)
    ax.axhline(0, color="0.15", lw=1.4)
    ax.set_xlim(0, max_step)
    return scol, hcol, missing


def main():
    cv = {**pt.curves(), **(pt.load(pt.DATA, "curves_tsweep_pnorm.json") or {})}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    for row, (K, tau) in enumerate(PANELS):
        J = pt.CELLS[K]
        for col, (arm, prefix, suffix) in enumerate(ARMS):
            ax = axes[row][col]
            scol, hcol, missing = panel(ax, cv, K, tau, prefix, suffix)
            for m in missing:
                print(f"no curve for {m}")
            ax.set_title(rf"$K={K}$, $J={J}$, $\tau={tau:g}$, {arm}")
            if row == 1:
                ax.set_xlabel("step")
            if col == 0:
                ax.set_ylabel(rf"val CE $-$ hard Top-{K}")
        axes[row][0].sharey(axes[row][1])
        axes[row][1].tick_params(labelleft=False)
        hd = [Line2D([0], [0], color="0.15", lw=1.4, label=f"hard Top-{K} (post-norm)")]
        hd += [Line2D([0], [0], color=hcol[kp], ls="--", lw=1.6, label=f"hard Top-{kp}") for kp in pt.EXTRA_HARD[K]]
        hd += [Line2D([0], [0], color=pt.GREY, ls=":", lw=1.8,
                      label=r"first-order, $T=2$, $\gamma=1$, no post-norm")]
        hd += [Line2D([0], [0], color=scol[s], lw=2.0, label=rf"$s={s:g}$") for _, s in pt.SS]
        axes[row][1].legend(handles=hd, loc="best", framealpha=0.93, ncol=2)
    fig.suptitle(r"First-order scope, span rule $T=\tau\,(s_{(K+1)}-s_{(K+J)})$, $\gamma=2sT/b$, "
                 "code carried: without (left) and with (right) the output RMSNorm", y=1.0, fontsize=12)
    fig.tight_layout()
    pt.save(fig, "fig_ts_fo_span_pnorm")
    print(os.path.join(pt.OUT, "fig_ts_fo_span_pnorm.png"))


if __name__ == "__main__":
    main()
