"""Encoder rows of the candidate pool, and the score decomposition.

s_i = ||w_i|| ||x|| |cos(w_i, x)|.  Two figures:

  rows_*    per-candidate bars against score at several steps: the row's
            learnable gain and its fused norm, boundary feature in red.
  decomp_*  the three factors over training, for both modes.

Usage: python plot_rows.py outdir/ npz_dir/
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

outdir, npzdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

runs = {}
for f in glob.glob(os.path.join(npzdir, "row_*.npz")):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    var = "kappa" if "_kappa_" in os.path.basename(f) else "nokappa"
    runs[((m["k"], m["j"], m["T"]), var)] = (d, m)

CFGS = sorted({c for c, _ in runs})
STEPS = [0, 300, 500, 700, 1000]
C_ACT, C_INACT, C_BND = "#0173B2", "#9EC9E2", "#CC3311"
COL = {"kappa": "#0173B2", "nokappa": "#CC3311"}

# ---- per-candidate row quantities ---------------------------------------- #
for cfg in CFGS:
    K, J, T = cfg
    for field, label, tag in (("row_gain", "row gain $\\mathrm{softplus}(\\rho_i)$", "gain"),
                              ("row_fused", "$\\|w_i\\|$", "norm")):
        fig, axes = plt.subplots(2, len(STEPS), figsize=(3.0 * len(STEPS), 5.4))
        for r, var in enumerate(("kappa", "nokappa")):
            d, _ = runs[(cfg, var)]
            st = d["steps"]
            for c, target in enumerate(STEPS):
                ax = axes[r][c]
                i = int(np.argmin(np.abs(st - target)))
                s, v = d["s"][i], d[field][i]
                w = max((s.max() - s.min()) / max(len(s), 1) * 0.8, 1e-12)
                ax.bar(s[:K], v[:K], width=w, color=C_ACT, linewidth=0)
                ax.bar(s[K + 1:], v[K + 1:], width=w, color=C_INACT, linewidth=0)
                ax.bar([s[K]], [v[K]], width=w * 3, color=C_BND, linewidth=0, zorder=5)
                lo, hi = float(v.min()), float(v.max())
                pad = 0.12 * (hi - lo) + 1e-6
                ax.set_ylim(lo - pad, hi + pad)
                ax.tick_params(labelsize=8)
                if r == 0:
                    ax.set_title(f"step {st[i]}", fontsize=11)
                if c == 0:
                    ax.set_ylabel(("kappa" if var == "kappa" else "no kappa")
                                  + f"\n{label}", fontsize=9)
                if r == 1:
                    ax.set_xlabel("feature score $s_i$", fontsize=9)
        h = [Line2D([0], [0], color=C_ACT, lw=6, label="active"),
             Line2D([0], [0], color=C_INACT, lw=6, label="inactive"),
             Line2D([0], [0], color=C_BND, lw=6, label="boundary")]
        axes[0][-1].legend(handles=h, fontsize=8, loc="best", framealpha=0.9)
        fig.suptitle(f"Encoder {label} of the top-$(K{{+}}J)$ candidates, "
                     f"$K={K}$, $J={J}$ (block 4, one token)", fontsize=12, y=0.995)
        fig.tight_layout()
        p = os.path.join(outdir, f"rows_{tag}_k{K}_j{J}.png")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(p)

# ---- the decomposition over training ------------------------------------- #
fig, axes = plt.subplots(len(CFGS), 4, figsize=(17.0, 4.0 * len(CFGS)))
axes = np.atleast_2d(axes)
for r, cfg in enumerate(CFGS):
    K, J, T = cfg
    for var in ("kappa", "nokappa"):
        d, _ = runs[(cfg, var)]
        st = d["steps"]
        s1 = d["s"][:, 0]
        wn = d["row_fused"].mean(-1)
        cs = d["cos"].mean(-1)
        for c, (y, lab) in enumerate((
                (s1, "top score $s_{(1)}$"),
                (d["x_norm"], "stream norm $\\|x\\|$"),
                (wn, "pool mean $\\|w_i\\|$"),
                (cs, "pool mean $|\\cos(w_i,x)|$"))):
            axes[r][c].plot(st, y, "-o", color=COL[var], ms=4, lw=2.0)
            axes[r][c].set_yscale("log")
            axes[r][c].grid(alpha=0.3)
            axes[r][c].set_xlabel("training step")
            axes[r][c].set_title(f"{lab}  ($K={K}$, $J={J}$)", fontsize=10)
h = [Line2D([0], [0], color=COL["kappa"], lw=2, marker="o", ms=4, label="kappa"),
     Line2D([0], [0], color=COL["nokappa"], lw=2, marker="o", ms=4, label="no kappa")]
axes[0][0].legend(handles=h, fontsize=9, loc="best")
fig.suptitle("Score decomposition $s_i=\\|w_i\\|\\,\\|x\\|\\,|\\cos(w_i,x)|$ "
             "over training (log axes)", fontsize=13, y=0.998)
fig.tight_layout()
p = os.path.join(outdir, "rows_decomposition.png")
fig.savefig(p, dpi=170, bbox_inches="tight")
print(p)

# ---- the accounting the text quotes -------------------------------------- #
print()
for cfg in CFGS:
    K, J, T = cfg
    for var in ("kappa", "nokappa"):
        d, _ = runs[(cfg, var)]
        f0, f1 = 0, -1
        def g(a):
            return float(a[f1]) / max(float(a[f0]), 1e-30)
        print("K=%-4d %-9s  s_1 x%-10.4g = ||x|| x%-8.4g * ||w|| x%-7.4f * |cos| x%-7.4f"
              " | pool gain %.4f -> %.4f | row||w|| pool %.4f -> %.4f (all rows %.4f -> %.4f)"
              % (K, var, g(d["s"][:, 0]), g(d["x_norm"]),
                 g(d["row_fused"].mean(-1)), g(d["cos"].mean(-1)),
                 d["row_gain"][0].mean(), d["row_gain"][-1].mean(),
                 d["row_fused"][0].mean(), d["row_fused"][-1].mean(),
                 d["all_row_fused_mean"][0], d["all_row_fused_mean"][-1]))
