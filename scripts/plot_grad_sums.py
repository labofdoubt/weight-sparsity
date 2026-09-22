"""Pool sums of the two gradients over training: kappa vs no-kappa.

Top row:    sum_i dL/dp_i over the top-(K+J) candidates (signed).
Bottom row: sum_i dL/ds_i (signed), against sum_i |dL/ds_i| for scale.

Both mode corrections remove the common mode, so the score-gradient sum is
zero by construction in either mode -- the bottom row is a check of that, and
the gap to the dashed curve is how complete the cancellation is.

Usage: python plot_sums.py outdir/ npz_dir/
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
    var = "kappa" if "_kappa_" in os.path.basename(f) else "nokappa"
    runs[((m["k"], m["j"], m["T"]), var)] = (d, m)

CFGS = sorted({c for c, _ in runs})
COL = {"kappa": "#0173B2", "nokappa": "#CC3311"}

fig, axes = plt.subplots(2, len(CFGS), figsize=(6.8 * len(CFGS), 8.2))
axes = np.atleast_2d(axes)
for c, cfg in enumerate(CFGS):
    K, J, T = cfg
    for var in ("kappa", "nokappa"):
        if (cfg, var) not in runs:
            continue
        d, _ = runs[(cfg, var)]
        st = d["steps"]
        gp = d["dLdp"][:, 0, 0]      # block 4, token 0
        gs = d["dLds"][:, 0, 0]
        axes[0][c].plot(st, gp.sum(-1), "-o", color=COL[var], ms=4, lw=2.0)
        axes[1][c].plot(st, gs.sum(-1), "-o", color=COL[var], ms=4, lw=2.0)
        axes[1][c].plot(st, np.abs(gs).sum(-1), "--", color=COL[var], lw=1.5,
                        alpha=0.8)
    for r, lab in enumerate((r"$\sum_i \partial L/\partial p_i$",
                             r"$\sum_i \partial L/\partial s_i$")):
        ax = axes[r][c]
        ax.axhline(0, color="0.35", lw=0.9)
        ax.set_yscale("symlog", linthresh=1e-12 if r else 1e-6)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
        if c == 0:
            ax.set_ylabel(lab)
        ax.set_title(f"$K={K}$, $J={J}$, $T={T:g}$", fontsize=11)
h = [Line2D([0], [0], color=COL["kappa"], lw=2.0, marker="o", ms=4, label="kappa"),
     Line2D([0], [0], color=COL["nokappa"], lw=2.0, marker="o", ms=4, label="no kappa"),
     Line2D([0], [0], color="0.3", lw=1.5, ls="--",
            label=r"$\sum_i |\partial L/\partial s_i|$ (term scale)")]
axes[1][-1].legend(handles=h, fontsize=9, loc="lower left", framealpha=0.93)
axes[0][-1].legend(handles=h[:2], fontsize=9, loc="lower left", framealpha=0.93)
fig.suptitle("Pool sums of the two gradients over training "
             "(block 4, one token; symlog axes)", fontsize=13, y=0.995)
fig.tight_layout()
p = os.path.join(outdir, "grad_sums.png")
fig.savefig(p, dpi=170, bbox_inches="tight")
print(p)

print()
for cfg in CFGS:
    for var in ("kappa", "nokappa"):
        if (cfg, var) not in runs:
            continue
        d, m = runs[(cfg, var)]
        gp = d["dLdp"][:, 0, 0].sum(-1)
        gs = d["dLds"][:, 0, 0]
        scale = np.abs(gs).sum(-1)
        alive = scale > 1e-20          # post-collapse every term underflows to 0
        rel = np.abs(gs.sum(-1))[alive] / scale[alive]
        print("K=%-4d J=%-4d %-9s  sum dL/dp in [%+.2e, %+.2e] (worst at step %d), "
              "%d/%d negative | relative |sum dL/ds| max %.1e over %d live steps"
              % (m["k"], m["j"], var, gp.min(), gp.max(),
                 d["steps"][int(np.abs(gp).argmax())], int((gp < 0).sum()),
                 len(gp), rel.max() if len(rel) else 0.0, int(alive.sum())))
