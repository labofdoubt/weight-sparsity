"""Stabilization comparison: five approaches per K group.

One figure per K (32, 64, 128): the smaller candidate window J on the left,
the larger on the right, with the vertical axis capped at 2.0 nats -- the arms
differ only below that, so the full-range view is omitted.  Colour
encodes the approach.  A cross marks where the divergence guard stopped a run.

Runs come from two boxes: the four T/lr/post-norm arms and the support-gradient
scale arm.

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

# (match token, colour, label, linestyle, linewidth, alpha) -- fixed order, so
# a colour always means the same intervention across all three figures.  The
# line style carries the grouping: the untouched baseline is dashed, the two
# arms that stabilize without an accuracy cost are solid, and the two
# attenuating arms are dash-dotted and slightly transparent.
ARMS = [("stab_base",  "#0173B2", "baseline: $T{=}1$, lr $6{\\times}10^{-4}$",
         "--",  2.0, 1.00),
        ("stab_t2",    "#DE8F05", "$T{=}2$",
         "-",   2.1, 1.00),
        ("stab_lr2e4", "#029E73", "lr $2{\\times}10^{-4}$",
         "-.",  1.9, 0.75),
        ("stab_pnorm", "#CC3311", "post-bottleneck RMSNorm",
         "-",   2.1, 1.00),
        ("supp03",     "#8856A7", "support grad scale $0.3$",
         "-.",  1.9, 0.75)]
YLIM_FULL = (1.40, 3.1)
YLIM_CAP = (1.40, 2.0)


STYLE = {a: (ls, lw, al) for a, _, _, ls, lw, al in ARMS}
COLOUR = {a: c for a, c, _, _, _, _ in ARMS}


def arm_of(name):
    for a, *_ in ARMS:
        if a in name:
            return a
    return None


def draw(ax, runs, ylim, xlabel):
    deaths, present = [], set()
    for name, rec in sorted(runs.items()):
        arm = arm_of(name)
        ls, lw, al = STYLE[arm]
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, ls, color=COLOUR[arm], lw=lw, alpha=al)
        present.add(arm)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], COLOUR[arm]))
    span = ylim[1] - ylim[0]
    for sx, vy, c in deaths:
        ax.plot(sx, min(vy, ylim[1] - 0.02 * span), "x", color=c, ms=11, mew=2.6)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    if xlabel:
        ax.set_xlabel("training step")
    return present, bool(deaths)


for K in (32, 64, 128):
    sel = {n: r for n, r in curves.items()
           if r.get("k") == K and r.get("val") and arm_of(n) and "hard" not in n}
    if not sel:
        print("no runs for K =", K)
        continue
    js = sorted({r["j"] for r in sel.values()})
    # one row only: the full-range view sat almost entirely above the region
    # the arms differ in, so it is dropped and the legend moves here.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    present, any_death = set(), False
    for c, J in enumerate(js):                       # left: small J, right: large J
        col = {n: r for n, r in sel.items() if r["j"] == J}
        p, d = draw(axes[c], col, YLIM_CAP, xlabel=True)
        present |= p
        any_death |= d
        axes[c].set_title(f"$J={J}$", fontsize=12)
    axes[0].set_ylabel("validation CE (nats), capped at 2.0")
    handles = [Line2D([0], [0], color=c, lw=lw, ls=ls, alpha=al, label=lab)
               for a, c, lab, ls, lw, al in ARMS if a in present]
    if any_death:
        handles.append(Line2D([0], [0], color="0.3", ls="", marker="x", ms=9,
                              mew=2.2, label="stopped as diverged"))
    axes[1].legend(handles=handles, fontsize=10, loc="upper right",
                   framealpha=0.93)
    fig.suptitle(f"RBLapSum through_rank_kappa, $K={K}$: five stabilizations",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"stab_k{K}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)
