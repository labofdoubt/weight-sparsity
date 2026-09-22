"""Signed per-candidate gradients against feature score: kappa vs no-kappa.

For one block and one token, bars of the signed gradient at each of the
top-(K+J) candidates, plotted against that candidate's score:

    dL/dp_i = u_i z_i            (u_i = dL/dy_i at the gate output)
    dL/ds_i = mode-corrected a_i,  a_i = (dL/dp_i) kappa_i

Descent moves a quantity *against* its gradient, so a negative bar means the
loss falls if that feature's gate variable (or score) increases.  Active
candidates (rank <= K) and inactive ones are shaded differently and the
boundary feature -- rank K+1, whose score defines b -- is drawn in red.

Usage: python plot_grad.py outdir/ npz_dir/
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
for f in glob.glob(os.path.join(npzdir, "grad_*.npz")):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    name = os.path.basename(f)[5:-4]
    variant = "kappa" if "_kappa_" in name else "nokappa"
    runs[((m["k"], m["j"], m["T"]), variant)] = (d, m)

CFGS = sorted({c for c, _ in runs})
STEPS = [0, 300, 500, 700, 1000]
C_ACT, C_INACT, C_BND = "#0173B2", "#9EC9E2", "#CC3311"
QTY = {"dLdp": ("$\\partial L/\\partial p_i$", "dLdp"),
       "dLds": ("$\\partial L/\\partial s_i$", "dLds")}


def panel(ax, s, g, K, first_col):
    """Signed bars at the candidate scores; boundary feature in red."""
    w = max((s.max() - s.min()) / max(len(s), 1) * 0.8, 1e-12)
    ax.bar(s[:K], g[:K], width=w, color=C_ACT, linewidth=0)
    if len(s) > K + 1:
        ax.bar(s[K + 1:], g[K + 1:], width=w, color=C_INACT, linewidth=0)
    ax.bar([s[K]], [g[K]], width=w * 3.0, color=C_BND, linewidth=0, zorder=5)
    ax.axhline(0, color="0.3", lw=0.8)
    lim = np.abs(g).max() * 1.15 + 1e-30
    ax.set_ylim(-lim, lim)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=8)
    ax.yaxis.get_offset_text().set_fontsize(7)


for cfg in CFGS:
    K, J, T = cfg
    if not all((cfg, v) in runs for v in ("kappa", "nokappa")):
        continue
    for key, (label, tag) in QTY.items():
        fig, axes = plt.subplots(2, len(STEPS), figsize=(3.0 * len(STEPS), 5.6))
        for r, var in enumerate(("kappa", "nokappa")):
            d, m = runs[(cfg, var)]
            steps = d["steps"]
            for c, target in enumerate(STEPS):
                ax = axes[r][c]
                i = int(np.argmin(np.abs(steps - target)))
                s = d["s"][i, 0, 0]          # prof layer 0 = block 4, token 0
                g = d[key][i, 0, 0]
                panel(ax, s, g, K, c == 0)
                if r == 0:
                    ax.set_title(f"step {steps[i]}", fontsize=11)
                if c == 0:
                    ax.set_ylabel(("kappa" if var == "kappa" else "no kappa")
                                  + f"\n{label}", fontsize=10)
                if r == 1:
                    ax.set_xlabel("feature score $s_i$", fontsize=9)
        h = [Line2D([0], [0], color=C_ACT, lw=6, label="active ($\\mathrm{rank}\\leq K$)"),
             Line2D([0], [0], color=C_INACT, lw=6, label="inactive"),
             Line2D([0], [0], color=C_BND, lw=6, label="boundary feature (rank $K{+}1$)")]
        axes[0][-1].legend(handles=h, fontsize=8, loc="upper right", framealpha=0.93)
        fig.suptitle(f"Signed {label} against score, $K={K}$, $J={J}$, $T={T:g}$ "
                     f"(block 4, one token)", fontsize=13, y=0.995)
        fig.tight_layout()
        p_out = os.path.join(outdir, f"grad_{tag}_k{K}_j{J}.png")
        fig.savefig(p_out, dpi=160, bbox_inches="tight")
        print(p_out)

# ---- the numbers the report quotes --------------------------------------- #
print()
print("%-30s %-9s %6s %12s %12s %12s %10s" % (
    "config", "mode", "step", "dLdp_bnd", "|dLds|_bnd", "|dLds|max_act",
    "frac inact dLdp<0"))
for cfg in CFGS:
    K, J, T = cfg
    for var in ("kappa", "nokappa"):
        if (cfg, var) not in runs:
            continue
        d, m = runs[(cfg, var)]
        st = d["steps"]
        for target in (0, 500, 1000):
            i = int(np.argmin(np.abs(st - target)))
            gp = d["dLdp"][i, 0, 0]; gs = d["dLds"][i, 0, 0]
            frac = float((gp[K:] < 0).mean())
            print("%-30s %-9s %6d %12.3g %12.3g %12.3g %10.2f" % (
                f"K={K},J={J},T={T:g}", var, st[i], gp[K], abs(gs[K]),
                np.abs(gs[:K]).max(), frac))
