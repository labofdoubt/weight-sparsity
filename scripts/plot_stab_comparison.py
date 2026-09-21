"""Stabilization comparison: five approaches per K group.

One figure per K (32, 64, 128).  Colour encodes the approach; line style
encodes J (solid = wide window K+J=512, dashed = narrow J=K).  A cross marks
where the divergence guard stopped a run.  Runs come from two boxes: the four
T/lr/post-norm arms (california) and the support-scale arm (korea).

Usage: python plot_stab.py outdir/ curves.json [curves2.json ...]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

outdir, jsons = sys.argv[1], sys.argv[2:]
os.makedirs(outdir, exist_ok=True)
curves = {}
for p in jsons:
    curves.update(json.load(open(p)))

# (match token, colour, label) -- fixed order, so a colour always means the
# same intervention across all three figures
ARMS = [("stab_base",  "#0173B2", "baseline: $T{=}1$, lr $6{\\times}10^{-4}$"),
        ("stab_t2",    "#DE8F05", "$T{=}2$"),
        ("stab_lr2e4", "#029E73", "lr $2{\\times}10^{-4}$"),
        ("stab_pnorm", "#CC3311", "post-bottleneck RMSNorm"),
        ("supp03",     "#8856A7", "support scale $0.3$")]
YLIM = (1.40, 3.1)


def arm_of(name):
    for a, _, _ in ARMS:
        if a in name:
            return a
    return None


for K in (32, 64, 128):
    sel = {n: r for n, r in curves.items()
           if r.get("k") == K and r.get("val") and arm_of(n)
           and "hard" not in n}
    if not sel:
        print("no runs for K =", K)
        continue
    js = sorted({r["j"] for r in sel.values()})
    styles = {js[0]: ("--", 1.7), js[-1]: ("-", 2.2)}
    colour = {a: c for a, c, _ in ARMS}
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    deaths, rows = [], []
    for name, rec in sorted(sel.items()):
        arm = arm_of(name)
        ls, lw = styles[rec["j"]]
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, ls, color=colour[arm], lw=lw)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], colour[arm]))
        rows.append((arm, rec["j"], v[-1], "diverged" in rec))
    for sx, vy, c in deaths:
        ax.plot(sx, min(vy, YLIM[1] - 0.03), "x", color=c, ms=11, mew=2.6)
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation CE (nats)")
    ax.set_title(f"RBLapSum through_rank_kappa, $K={K}$: five stabilizations",
                 fontsize=13)
    arm_handles = [Line2D([0], [0], color=c, lw=2.4, label=lab)
                   for a, c, lab in ARMS if any(r[0] == a for r in rows)]
    j_handles = [Line2D([0], [0], color="0.3", ls=styles[j][0], lw=styles[j][1],
                        label=f"$J={j}$") for j in js]
    if deaths:
        j_handles.append(Line2D([0], [0], color="0.3", ls="", marker="x", ms=9,
                                mew=2.2, label="stopped as diverged"))
    leg = ax.legend(handles=arm_handles, fontsize=10, loc="upper right",
                    framealpha=0.93)
    ax.add_artist(leg)
    ax.legend(handles=j_handles, fontsize=10, loc="lower left", framealpha=0.93)
    fig.tight_layout()
    p = os.path.join(outdir, f"stab_k{K}.png")
    fig.savefig(p, dpi=170)
    print(p)
    for arm, j, v, dv in sorted(rows):
        print("    %-11s J=%-4d %s" % (arm, j, "DIVERGED" if dv else "%.4f" % v))
