"""Stabilization comparison: five approaches per K group.

One figure per K (32, 64, 128), each a 2x2 grid: the smaller candidate window
J on the left, the larger on the right; the top row over the full range and the
bottom row the same panels with the vertical axis capped at 2.0 nats.  Colour
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

# (match token, colour, label) -- fixed order, so a colour always means the
# same intervention across all three figures
ARMS = [("stab_base",  "#0173B2", "baseline: $T{=}1$, lr $6{\\times}10^{-4}$"),
        ("stab_t2",    "#DE8F05", "$T{=}2$"),
        ("stab_lr2e4", "#029E73", "lr $2{\\times}10^{-4}$"),
        ("stab_pnorm", "#CC3311", "post-bottleneck RMSNorm"),
        ("supp03",     "#8856A7", "support grad scale $0.3$")]
YLIM_FULL = (1.40, 3.1)
YLIM_CAP = (1.40, 2.0)


def arm_of(name):
    for a, _, _ in ARMS:
        if a in name:
            return a
    return None


def draw(ax, runs, ylim, xlabel):
    colour = {a: c for a, c, _ in ARMS}
    deaths, present = [], set()
    for name, rec in sorted(runs.items()):
        arm = arm_of(name)
        s = [p[0] for p in rec["val"]]
        v = [p[1] for p in rec["val"]]
        ax.plot(s, v, "-", color=colour[arm], lw=2.0)
        present.add(arm)
        if "diverged" in rec:
            deaths.append((s[-1], v[-1], colour[arm]))
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
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex="col", sharey="row")
    present, any_death = set(), False
    for c, J in enumerate(js):                       # left: small J, right: large J
        col = {n: r for n, r in sel.items() if r["j"] == J}
        for row, ylim in enumerate((YLIM_FULL, YLIM_CAP)):
            p, d = draw(axes[row][c], col, ylim, xlabel=(row == 1))
            present |= p
            any_death |= d
        axes[0][c].set_title(f"$J={J}$", fontsize=12)
    axes[0][0].set_ylabel("validation CE (nats)")
    axes[1][0].set_ylabel("validation CE (nats), capped at 2.0")
    handles = [Line2D([0], [0], color=c, lw=2.4, label=lab)
               for a, c, lab in ARMS if a in present]
    if any_death:
        handles.append(Line2D([0], [0], color="0.3", ls="", marker="x", ms=9,
                              mew=2.2, label="stopped as diverged"))
    axes[0][1].legend(handles=handles, fontsize=10, loc="upper right",
                      framealpha=0.93)
    fig.suptitle(f"RBLapSum through_rank_kappa, $K={K}$: five stabilizations",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"stab_k{K}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)
