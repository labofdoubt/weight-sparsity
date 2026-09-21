"""Heatmaps of the initialization surrogate gain Pi over the (K, J) grid.

Page 1: 3x3 grid of log-scaled Pi heatmaps -- rows are kernel temperatures,
columns are blocks (early / middle / deep), so one row compares depth at fixed
T and one column compares T at fixed depth.  Page 2: the boundary geometry b and
the local rank spacing delta, which depend on K only (not on J or T), then exact Pi
against the dense-boundary proxy b^2/(4 T delta) at T = 1, once for the widest
candidate window and once for the narrowest, on one shared vertical range.

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


def grid(T, blk, field="Pi_med"):
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
    fig, axes = plt.subplots(len(Ts), len(blocks), figsize=(14.6, 13.2))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for r, T in enumerate(Ts):           # rows: temperature
        for c, blk in enumerate(blocks):  # columns: depth
            ax = axes[r][c]
            M = mats[(T, blk)]
            im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis",
                           norm=norm,
                           extent=[Js[0] - 16, Js[-1] + 16, Ks[0] - 16, Ks[-1] + 16])
            # iso-contours help read the shape off a log colour map
            cs = ax.contour(Js, Ks, M, levels=[100, 250, 500, 1000, 2000, 5000],
                            colors="0.15", linewidths=0.8, alpha=0.8)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
            ax.set_title(f"block {blk} ({BLOCK_LABEL.get(blk, '')}), $T={T:g}$",
                         fontsize=11)
            ax.set_xticks(Js[1::2])
            ax.set_yticks(Ks[1::2])
            ax.tick_params(labelsize=8)
            if r == len(Ts) - 1:
                ax.set_xlabel("$J$ (candidate window)")
            if c == 0:
                ax.set_ylabel("$K$ (active features)")
    fig.suptitle("Surrogate gain $\\Pi=\\|L_\\kappa D_z\\|_F^2$ at initialization, "
                 "rblapsum through_rank_kappa ($b_0=0$)", fontsize=14, y=0.995)
    cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02, aspect=30)
    cb.set_label("$\\Pi$ at step 0 (token median, log scale)", fontsize=11)
    if png_prefix:
        fig.savefig(png_prefix + "_heatmaps.png", dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- page 2: b, delta, Pi_approx ------------------------------------- #
    fig, axes = plt.subplots(1, 4, figsize=(19.4, 4.4))
    colors = {blk: c for blk, c in zip(blocks, ["#0173B2", "#DE8F05", "#029E73"])}
    for blk in blocks:
        b = [d["boundary"][f"K{K}_blk{blk}"]["b_mean"] for K in Ks]
        dl = [d["boundary"][f"K{K}_blk{blk}"]["delta_mean"] for K in Ks]
        lab = f"block {blk} ({BLOCK_LABEL.get(blk, '')})"
        axes[0].plot(Ks, b, "-o", color=colors[blk], ms=4, label=lab)
        axes[1].plot(Ks, dl, "-o", color=colors[blk], ms=4, label=lab)
    axes[0].set_ylabel("$b$ at step 0")
    axes[0].set_title("boundary $b=s_{(K+1)}$ (token mean)")
    axes[1].set_ylabel("$\\delta$ at step 0")
    axes[1].set_title("rank spacing $\\delta=(s_{(K-4)}-s_{(K+4)})/8$ (token mean)")
    axes[1].set_yscale("log")
    T_REF = 1.0
    # Panels 3 and 4: exact Pi (solid) against the dense-boundary proxy
    # (dashed) at a wide and a narrow candidate window.  Pi_approx has no J
    # dependence by construction -- the same curve appears in both panels --
    # so the J=32 panel shows where the proxy stops applying.
    #
    # Both curves are token MEDIANS.  The proxy has no usable mean at init
    # (delta -> 0 on near-tied ranks sends b^2/(4 T delta) to 1e9-1e11), and
    # comparing a mean against a median would not be like for like; exact Pi's
    # own mean and median agree to within 1%, so the median costs nothing.
    for ax, J_REF in ((axes[2], Js[-1]), (axes[3], Js[0])):
        for blk in blocks:
            pi = [d["grid"][f"K{K}_J{J_REF}_T{T_REF:g}_blk{blk}"]["Pi_med"]
                  for K in Ks]
            pa = [d["approx"][f"K{K}_T{T_REF:g}_blk{blk}"]["Pi_approx_med"]
                  for K in Ks]
            ax.plot(Ks, pi, "-o", color=colors[blk], ms=4)
            ax.plot(Ks, pa, "--", color=colors[blk], lw=1.4, alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylabel("$\\Pi$ at step 0 (token median)")
        ax.set_title(f"$\\Pi$ and $\\Pi_{{\\rm approx}}$ at "
                     f"$T={T_REF:g}$, $J={J_REF}$")

    from matplotlib.lines import Line2D
    blk_handles = [Line2D([0], [0], color=colors[blk], lw=2.0, marker="o", ms=4,
                          label=f"block {blk} ({BLOCK_LABEL.get(blk, '')})")
                   for blk in blocks]
    kind_handles = [Line2D([0], [0], color="0.3", ls="-", marker="o", ms=4,
                           label="exact $\\Pi$"),
                    Line2D([0], [0], color="0.3", ls="--", lw=1.4,
                           label="$b^2/(4T\\delta)$")]
    for ax in (axes[2], axes[3]):
        leg = ax.legend(handles=blk_handles, fontsize=9, loc="lower right")
        ax.add_artist(leg)
        ax.legend(handles=kind_handles, fontsize=9, loc="upper left")
    # one shared vertical range makes the two panels directly comparable
    lo = min(axes[2].get_ylim()[0], axes[3].get_ylim()[0])
    hi = max(axes[2].get_ylim()[1], axes[3].get_ylim()[1])
    axes[2].set_ylim(lo, hi)
    axes[3].set_ylim(lo, hi)
    for ax in axes:
        ax.set_xlabel("$K$")
        ax.grid(alpha=0.3)

    fig.suptitle("Boundary geometry and surrogate gain at initialization "
                 "($b$ and $\\delta$ depend on $K$ only, not on $J$ or $T$)",
                 fontsize=13, y=1.02)
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
