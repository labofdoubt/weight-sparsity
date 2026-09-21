"""Validation-curve figures for the kappa-correction ablation (b0 = 0).

One figure per K group -- (K=32, T=1), (K=64, T=1), (K=128, T=2) -- each
comparing through_rank_kappa (solid) with plain through_rank (dashed) under
full MD.  Color encodes J; a cross marks the point where a run was stopped
as diverged, clamped to the top edge when the curve left the range.

Usage: python kappa_figures.py mdinit_curves.json outdir/
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

curves = json.load(open(sys.argv[1]))
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

J_COLORS = ["#0173B2", "#DE8F05"]
MODES = {  # mode -> (linestyle, lw)
    "kappa": ("-", 2.2),
    "through_rank": ("--", 1.8),
}
MODE_LABEL = {"kappa": "through_rank_kappa",
              "through_rank": "through_rank (no kappa)"}
YLIM = (1.39, 3.1)

GROUPS = [
    (32, 1.0, "k32", "$K=32$, $T=1$"),
    (64, 1.0, "k64", "$K=64$, $T=1$"),
    (128, 2.0, "k128_t2", "$K=128$, $T=2$"),
]

for K, T, tag, title in GROUPS:
    sel = {}
    for n, r in curves.items():
        if "_b00_" not in n or r["k"] != K or r.get("T", 1.0) != T:
            continue
        mode = "kappa" if "rblapsum_kappa" in n else "through_rank"
        sel[n] = (r, mode)
    if not sel:
        continue
    js = sorted({r["j"] for r, _ in sel.values()})
    colors = {j: J_COLORS[i] for i, j in enumerate(js)}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    deaths = []
    for n, (r, mode) in sorted(sel.items()):
        ls, lw = MODES[mode]
        col = colors[r["j"]]
        s = [p[0] for p in r["val"]]
        v = [p[1] for p in r["val"]]
        ax.plot(s, v, ls, color=col, lw=lw)
        if "diverged" in r:
            deaths.append((s[-1], v[-1], col))
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 20000)
    for sx, vy, col in deaths:
        ax.plot(sx, min(vy, YLIM[1] - 0.03), "x", color=col, ms=10, mew=2.4)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation CE (nats)")
    ax.set_title("RBLapSum kappa ablation, %s, $b_0=0$" % title, fontsize=13)
    handles = [Line2D([0], [0], color=colors[j], lw=2.6, label="$J=%d$" % j)
               for j in js]
    handles += [Line2D([0], [0], color="0.25", ls=MODES[m][0], lw=MODES[m][1],
                       label=MODE_LABEL[m]) for m in MODES]
    if deaths:
        handles.append(Line2D([0], [0], color="0.25", ls="", marker="x", ms=8,
                              mew=2.0, label="stopped as diverged"))
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.92)
    fig.tight_layout()
    p = os.path.join(outdir, "kappa_val_%s.png" % tag)
    fig.savefig(p, dpi=170)
    print(p)
