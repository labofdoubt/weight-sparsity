"""p_i profiles and surrogate mass over training: kappa vs no-kappa.

p_i = F((s_i - b)/T), F the Laplace CDF, b = max(b0, s_(K+1)).  Two figure
kinds per configuration:

  profile   bars of p_i against the feature's score, one panel per measured
            step, kappa on the top row and plain through_rank below; the red
            line is the K / K+1 boundary.
  mass      sum_i p_i against training step, with the part carried by the
            inactive candidates (ranks > K) shown separately.

Usage: python plot_mass.py outdir/ npz_dir/
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
for f in glob.glob(os.path.join(npzdir, "*.npz")):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    name = os.path.basename(f)[:-4]
    variant = "kappa" if "_kappa_" in name else "nokappa"
    cfg = (m["k"], m["j"], m["T"])
    runs[(cfg, variant)] = (d, m)

CFGS = sorted({c for c, _ in runs})
PROF_STEPS = [0, 300, 500, 700, 1000, 2500]
VAR_LABEL = {"kappa": "through\\_rank\\_kappa", "nokappa": "through\\_rank (no kappa)"}
COL = {"kappa": "#0173B2", "nokappa": "#CC3311"}

# ---- profiles ------------------------------------------------------------ #
for cfg in CFGS:
    K, J, T = cfg
    have = [v for (c, v) in runs if c == cfg]
    if len(have) < 2:
        continue
    fig, axes = plt.subplots(2, len(PROF_STEPS), figsize=(3.0 * len(PROF_STEPS), 6.2))
    for r, var in enumerate(("kappa", "nokappa")):
        d, m = runs[(cfg, var)]
        steps = d["steps"]
        for c, target in enumerate(PROF_STEPS):
            ax = axes[r][c]
            i = int(np.argmin(np.abs(steps - target)))
            s = d["prof_s"][i, 0, 0]          # prof layer 0 = block 4, token 0
            p = d["prof_p"][i, 0, 0]
            ax.vlines(s, 0, p, color=COL[var], lw=0.7, alpha=0.85)
            b = s[K]
            ax.axvline(b, color="red", lw=1.4)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(max(0, s.min() - 0.02 * (s.max() - s.min() + 1e-9)),
                        s.max() * 1.02 + 1e-9)
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(f"step {steps[i]}", fontsize=11)
            if c == 0:
                ax.set_ylabel(("kappa" if var == "kappa" else "no kappa")
                              + "\n$p_i$", fontsize=10)
            ax.text(0.97, 0.06, "$\\sum p_i$=%.0f" % p.sum(), transform=ax.transAxes,
                    ha="right", fontsize=8, color="0.25")
            if r == 1:
                ax.set_xlabel("feature score $s_i$", fontsize=9)
    fig.suptitle(f"$p_i$ over the candidate pool, $K={K}$, $J={J}$, $T={T:g}$ "
                 f"(block 4, one token; red line = $K/K{{+}}1$ boundary)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    p_out = os.path.join(outdir, f"mass_profile_k{K}_j{J}.png")
    fig.savefig(p_out, dpi=160, bbox_inches="tight")
    print(p_out)

# ---- mass vs step -------------------------------------------------------- #
fig, axes = plt.subplots(1, len(CFGS), figsize=(6.6 * len(CFGS), 4.6))
axes = np.atleast_1d(axes)
for a, cfg in enumerate(CFGS):
    K, J, T = cfg
    ax = axes[a]
    for var in ("kappa", "nokappa"):
        if (cfg, var) not in runs:
            continue
        d, m = runs[(cfg, var)]
        steps = d["steps"]
        tot = d["prof_p"][:, 0, 0, :].sum(-1)
        inact = d["prof_p"][:, 0, 0, K:].sum(-1)
        ax.plot(steps, tot / K, "-", color=COL[var], lw=2.2)
        ax.plot(steps, inact / K, "--", color=COL[var], lw=1.6, alpha=0.8)
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("$\\sum_i p_i \;/\; K$")
    ax.set_title(f"$K={K}$, $J={J}$, $T={T:g}$", fontsize=12)
    ax.grid(alpha=0.3)
h = [Line2D([0], [0], color=COL["kappa"], lw=2.2, label="kappa"),
     Line2D([0], [0], color=COL["nokappa"], lw=2.2, label="no kappa"),
     Line2D([0], [0], color="0.3", lw=2.0, label="total mass"),
     Line2D([0], [0], color="0.3", lw=1.6, ls="--", label="inactive part (ranks $>K$)"),
     Line2D([0], [0], color="0.4", lw=1.0, ls=":", label="mass $=K$")]
axes[-1].legend(handles=h, fontsize=9, loc="lower left", framealpha=0.93)
fig.suptitle("Surrogate mass over training (block 4, one token; "
             "normalized by $K$, log scale)", fontsize=13, y=1.0)
fig.tight_layout()
p_out = os.path.join(outdir, "mass_vs_step.png")
fig.savefig(p_out, dpi=170, bbox_inches="tight")
print(p_out)

# ---- the numbers the report quotes --------------------------------------- #
print()
for cfg in CFGS:
    K, J, T = cfg
    for var in ("kappa", "nokappa"):
        if (cfg, var) not in runs:
            continue
        d, m = runs[(cfg, var)]
        st = d["steps"]
        tot = d["prof_p"][:, 0, 0, :].sum(-1)
        ina = d["prof_p"][:, 0, 0, K:].sum(-1)
        smax = d["prof_s"][:, 0, 0, 0]
        print("K=%-4d J=%-4d %-9s tot/K %.2f -> %.2f | inactive/K %.2f -> %.3f | "
              "s_1 %.1f -> %.3g | CE %.2f -> %.2f"
              % (K, J, var, tot[0] / K, tot[-1] / K, ina[0] / K, ina[-1] / K,
                 smax[0], smax[-1], d["ce"][0], d["ce"][-1]))
