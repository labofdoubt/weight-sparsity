"""Heatmaps of the initialization surrogate gain Pi over the (K, J) grid.

Page 1: 3x3 grid of log-scaled Pi heatmaps -- rows are blocks (early / middle
/ deep), columns are kernel temperatures.  Page 2: the boundary geometry b and
the local rank spacing delta, which depend on K only (not on J or T), plus the
dense-boundary proxy Pi_approx = b^2/(4 T delta).

Usage: python plot_init_pi.py init_pi_grid.json out.pdf [out_png_prefix]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

d = json.load(open(sys.argv[1]))
out_pdf = sys.argv[2]
png_prefix = sys.argv[3] if len(sys.argv) > 3 else None

meta = d["meta"]
Ks = meta["k_values"]
Js = meta["j_values"]
Ts = meta["temps"]
blocks = meta["blocks"]
BLOCK_LABEL = {blocks[0]: "early", blocks[len(blocks) // 2]: "middle",
               blocks[-1]: "deep"}


def grid(T, blk, field="Pi_mean"):
    M = np.full((len(Ks), len(Js)), np.nan)
    for i, K in enumerate(Ks):
        for j, J in enumerate(Js):
            rec = d["grid"].get(f"K{K}_J{J}_T{T:g}_blk{blk}")
            if rec is not None:
                M[i, j] = rec[field]
    return M


mats = {(T, b): grid(T, b) for T in Ts for b in blocks}
finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mats.values()])
vmin, vmax = float(finite.min()), float(finite.max())

with PdfPages(out_pdf) as pdf:
    # ---- page 1: the 9 heatmaps ------------------------------------------ #
    fig, axes = plt.subplots(len(blocks), len(Ts), figsize=(14.6, 13.2))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for r, blk in enumerate(blocks):
        for c, T in enumerate(Ts):
            ax = axes[r][c]
            M = mats[(T, blk)]
            im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma",
                           norm=norm,
                           extent=[Js[0] - 16, Js[-1] + 16, Ks[0] - 16, Ks[-1] + 16])
            # iso-contours help read the shape off a log colour map
            cs = ax.contour(Js, Ks, M, levels=[100, 250, 500, 1000, 2000, 5000],
                            colors="white", linewidths=0.7, alpha=0.75)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
            ax.set_title(f"block {blk} ({BLOCK_LABEL.get(blk, '')}), $T={T:g}$",
                         fontsize=11)
            ax.set_xticks(Js[1::2])
            ax.set_yticks(Ks[1::2])
            ax.tick_params(labelsize=8)
            if r == len(blocks) - 1:
                ax.set_xlabel("$J$ (candidate window)")
            if c == 0:
                ax.set_ylabel("$K$ (active features)")
    fig.suptitle("Surrogate gain $\\Pi=\\|L_\\kappa D_z\\|_F^2$ at initialization, "
                 "rblapsum through_rank_kappa ($b_0=0$)", fontsize=14, y=0.995)
    cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02, aspect=30)
    cb.set_label("$\\Pi$ at step 0 (token mean, log scale)", fontsize=11)
    if png_prefix:
        fig.savefig(png_prefix + "_heatmaps.png", dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- page 2: b, delta, Pi_approx ------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4))
    colors = {blk: c for blk, c in zip(blocks, ["#0173B2", "#DE8F05", "#029E73"])}
    for blk in blocks:
        b = [d["boundary"][f"K{K}_blk{blk}"]["b_mean"] for K in Ks]
        bmed = [d["boundary"][f"K{K}_blk{blk}"]["b_med"] for K in Ks]
        dl = [d["boundary"][f"K{K}_blk{blk}"]["delta_mean"] for K in Ks]
        dlmed = [d["boundary"][f"K{K}_blk{blk}"]["delta_med"] for K in Ks]
        lab = f"block {blk} ({BLOCK_LABEL.get(blk, '')})"
        axes[0].plot(Ks, b, "-o", color=colors[blk], ms=4, label=lab)
        axes[0].plot(Ks, bmed, "--", color=colors[blk], lw=1, alpha=0.7)
        axes[1].plot(Ks, dl, "-o", color=colors[blk], ms=4, label=lab)
        axes[1].plot(Ks, dlmed, "--", color=colors[blk], lw=1, alpha=0.7)
    axes[0].set_ylabel("$b$ at step 0")
    axes[0].set_title("boundary $b=s_{(K+1)}$  (solid mean, dashed median)")
    axes[1].set_ylabel("$\\delta$ at step 0")
    axes[1].set_title("rank spacing $\\delta=(s_{(K-4)}-s_{(K+4)})/8$")
    axes[1].set_yscale("log")
    for T in Ts:
        for blk in blocks:
            pa = [d["approx"][f"K{K}_T{T:g}_blk{blk}"]["Pi_approx_med"] for K in Ks]
            axes[2].plot(Ks, pa, "-" if blk == blocks[0] else
                         ("--" if blk == blocks[1] else ":"),
                         color={1.0: "#0173B2", 2.0: "#DE8F05",
                                3.0: "#029E73"}.get(T, "0.4"),
                         lw=1.6, label=f"$T={T:g}$" if blk == blocks[0] else None)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("$b^2/(4T\\delta)$ at step 0 (median)")
    axes[2].set_title("proxy $\\Pi_{\\rm approx}$")
    for ax in axes[:2]:
        ax.legend(fontsize=9)
    from matplotlib.lines import Line2D
    t_handles = [Line2D([0], [0], color={1.0: "#0173B2", 2.0: "#DE8F05",
                                         3.0: "#029E73"}.get(T, "0.4"),
                        lw=1.8, label=f"$T={T:g}$") for T in Ts]
    b_handles = [Line2D([0], [0], color="0.3", ls=ls, lw=1.6,
                        label=f"block {blk} ({BLOCK_LABEL.get(blk, '')})")
                 for blk, ls in zip(blocks, ["-", "--", ":"])]
    leg_t = axes[2].legend(handles=t_handles, fontsize=9, loc="upper left")
    axes[2].add_artist(leg_t)
    axes[2].legend(handles=b_handles, fontsize=9, loc="lower right")
    for ax in axes:
        ax.set_xlabel("$K$")
        ax.grid(alpha=0.3)
    fig.suptitle("Boundary geometry at initialization ($b$ and $\\delta$ depend on "
                 "$K$ only, not on $J$ or $T$)", fontsize=13, y=1.02)
    fig.tight_layout()
    if png_prefix:
        fig.savefig(png_prefix + "_geometry.png", dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("wrote", out_pdf, "  Pi range %.3g .. %.3g" % (vmin, vmax))
for blk in blocks:
    row = [mats[(Ts[0], blk)][i, -1] for i in (0, len(Ks) // 2, len(Ks) - 1)]
    print("  block %d, T=%g, J=%d: Pi at K=%d/%d/%d = %.3g / %.3g / %.3g"
          % (blk, Ts[0], Js[-1], Ks[0], Ks[len(Ks)//2], Ks[-1], *row))
