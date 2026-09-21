"""Stabilization comparison: four approaches per K group.

One figure per K (32, 64, 128).  Color encodes the approach -- baseline
(T=1, lr=6e-4), T=2, lr=2e-4, and a post-bottleneck RMSNorm -- and line style
encodes J (solid = the wide window K+J=512, dashed = the narrow J=K).  A cross
marks the point where the divergence guard stopped a run.

Usage: python plot_stab.py curves.json outdir/
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

curves = json.load(open(sys.argv[1]))
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

ARMS = [("base", "#0173B2", "baseline: $T{=}1$, lr $6\\times10^{-4}$"),
        ("t2", "#DE8F05", "$T{=}2$"),
        ("lr2e4", "#029E73", "lr $2\\times10^{-4}$"),
        ("pnorm", "#CC3311", "post-bottleneck RMSNorm")]
YLIM = (1.40, 3.1)


def arm_of(name):
    for a, _, _ in ARMS:
        if f"_stab_{a}_" in name:
            return a
    return None


for K in (32, 64, 128):
    sel = {n: r for n, r in curves.items()
           if "_stab_" in n and r.get("k") == K and r["val"]}
    if not sel:
        print("no runs yet for K =", K)
        continue
    js = sorted({r["j"] for r in sel.values()})
    styles = {js[0]: ("--", 1.6), js[-1]: ("-", 2.1)}   # narrow dashed, wide solid
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    deaths, rows = [], []
    for name, rec in sorted(sel.items()):
        arm = arm_of(name)
        color = dict((a, c) for a, c, _ in ARMS)[arm]
        ls, lw = styles[rec["j"]]
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, ls, color=color, lw=lw)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], color))
        rows.append((arm, rec["j"], v[-1], s[-1], "diverged" in rec))
    for sx, vy, c in deaths:
        ax.plot(sx, min(vy, YLIM[1] - 0.03), "x", color=c, ms=10, mew=2.4)
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation CE (nats)")
    ax.set_title(f"RBLapSum through_rank_kappa, $K={K}$: four stabilizations",
                 fontsize=13)
    arm_handles = [Line2D([0], [0], color=c, lw=2.4, label=lab)
                   for a, c, lab in ARMS if any(r[0] == a for r in rows)]
    j_handles = [Line2D([0], [0], color="0.3", ls=styles[j][0], lw=styles[j][1],
                        label=f"$J={j}$") for j in js]
    if deaths:
        j_handles.append(Line2D([0], [0], color="0.3", ls="", marker="x", ms=8,
                                mew=2.0, label="stopped as diverged"))
    leg = ax.legend(handles=arm_handles, fontsize=10, loc="upper right",
                    framealpha=0.93)
    ax.add_artist(leg)
    ax.legend(handles=j_handles, fontsize=10, loc="lower left", framealpha=0.93)
    fig.tight_layout()
    p = os.path.join(outdir, f"stab_k{K}.png")
    fig.savefig(p, dpi=170)
    print(p)
    for arm, j, v, s, dv in sorted(rows):
        print("    %-8s J=%-4d %s" % (arm, j,
              f"diverged at {s}" if dv else f"final {v:.4f}"))
