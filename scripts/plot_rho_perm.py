"""Permutation ablation: val curves by J, rho as a colour ramp.

One panel per J (K=32 throughout, T=2, no post-norm): rho in a light->dark
ramp, the california ancestor run (rho = 0 by definition, trained on the
retired machine) as a black dashed reference.  The j32/j96 cells were stopped
early by hand; their curves end where they end.

Usage: python plot_rho.py out/ rho_curves.json sfcmp_curves.json
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

outdir, rho_p, ref_p = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
rho = json.load(open(rho_p))
refs = json.load(open(ref_p))

JS = [32, 96, 224, 480]
RHOS = [0.0, 0.25, 0.5, 1.0]
COL = {0.0: "#C6DBEF", 0.25: "#6BAED6", 0.5: "#2171B5", 1.0: "#08306B"}

def ancestor(j):
    for r in refs.values():
        if (r["mode"] == "rblapsum" and r["k"] == 32 and r["j"] == j
                and r["T"] == 2.0 and not r["pn"]):
            return r

fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharex=True, sharey=True)
for ax, j in zip(axes.ravel(), JS):
    a = ancestor(j)
    if a:
        s, v = zip(*a["val"])
        ax.plot(s, v, ls="--", color="black", lw=1.8, zorder=3)
    for r in sorted((r for r in rho.values() if r["j"] == j), key=lambda r: r["rho"]):
        if not r["val"]:
            continue
        s, v = zip(*r["val"])
        ax.plot(s, v, color=COL[r["rho"]], lw=2.0)
        if s[-1] < 19000:
            ax.plot([s[-1]], [min(v[-1], 2.28)], marker="|", color=COL[r["rho"]],
                    ms=12, mew=2.5, clip_on=False, zorder=5)
    ax.set_xlim(0, 20000)
    ax.set_ylim(1.4, 2.3)
    ax.grid(alpha=0.3)
    ax.set_title(f"$K=32$, $J={j}$", fontsize=12)
h = [Line2D([0], [0], ls="--", color="black", lw=1.8,
            label="ancestor run (california, $\\rho=0$)")]
h += [Line2D([0], [0], color=COL[r], lw=2.2, label=f"$\\rho={r:g}$") for r in RHOS]
axes[0][0].legend(handles=h, fontsize=9, loc="upper right")
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("validation CE (nats)")
fig.suptitle("Permuting a $\\rho$ fraction of the surrogate signal $dL/dp$ "
             "(rblapsum kappa, $T=2$): quality is monotone in the intact fraction",
             fontsize=13, y=0.997)
fig.tight_layout()
p = os.path.join(outdir, "rho_perm_by_j.png")
fig.savefig(p, dpi=170, bbox_inches="tight")
print(p)
