"""Simplified surrogate gain b^2/(4 T delta) against training step and block.

One panel per run: x = training step, one curve per block, coloured light to
dark with depth.  b = max(b0, s_(K+1)) and delta are read from the stored
ladder geometry, so nothing is recomputed from checkpoints.

Usage: python plot_approx_gain.py outdir/ geom_dir/ [b0]
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

outdir, geomdir = sys.argv[1], sys.argv[2]
B0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1   # the era's abs_topk default
os.makedirs(outdir, exist_ok=True)

# one record per (run, source); later sources fill in rungs the earlier lack
runs = {}
for src in ("kstab_pi", "kmap2", "brst"):
    p = os.path.join(geomdir, src, "ladder_geometry.json")
    if not os.path.exists(p):
        continue
    for name, r in json.load(open(p)).items():
        K, T = int(r["k"]), float(r["T"])
        per = runs.setdefault((name, K, int(r["j"]), T), {})
        for li, ckpts in enumerate(r["layers"]):
            for rec in ckpts:
                b = max(B0, float(rec["own"]["sK1"]))
                dl = float(rec["own"]["delta"])
                if dl <= 0:
                    continue                      # tied ranks: proxy undefined
                per.setdefault(li, {})[int(rec["step"])] = b * b / (4.0 * T * dl)

keys = sorted(runs, key=lambda t: (t[1], t[2], t[3]))
print("runs with usable geometry:", len(keys))
cmap = cm.get_cmap("viridis")
ncol = 4
nrow = int(np.ceil(len(keys) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                         sharey=True)
axes = np.atleast_2d(axes).ravel()
for ax, key in zip(axes, keys):
    name, K, J, T = key
    per = runs[key]
    for li in sorted(per):
        st = sorted(per[li])
        ax.plot(st, [per[li][s] for s in st], "-o", ms=2.5, lw=1.4,
                color=cmap(0.85 - 0.75 * li / 7))
    ax.set_yscale("log")
    # the axis is truncated at 1e4: the collapsing runs reach 1e10-1e16, which
    # would compress every surviving run onto a flat line.  Curves that leave
    # the top of a panel are diverging.
    ax.set_ylim(10, 1e4)
    ax.grid(alpha=0.3)
    ax.set_title(f"$K={K}$, $J={J}$, $T={T:g}$", fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes[len(keys):]:
    ax.axis("off")
for i, ax in enumerate(axes[:len(keys)]):
    if i % ncol == 0:
        ax.set_ylabel("$b^2/(4T\\delta)$")
    if i >= len(keys) - ncol:
        ax.set_xlabel("training step")
h = [Line2D([0], [0], color=cmap(0.85 - 0.75 * li / 7), lw=2,
            label=f"block {li}") for li in range(8)]
fig.legend(handles=h, fontsize=9, loc="lower center", ncol=8,
           bbox_to_anchor=(0.5, -0.012))
fig.suptitle("Simplified surrogate gain $b^2/(4T\\delta)$ by training step and "
             "block, per run (axis truncated at $10^4$)", fontsize=14, y=1.0)
fig.tight_layout()
p = os.path.join(outdir, "approx_gain_by_step_block.png")
fig.savefig(p, dpi=160, bbox_inches="tight")
print(p)

print()
print("%-40s %6s %8s %10s %10s" % ("run", "blocks", "ckpts", "gain@first", "gain@last"))
for key in keys:
    name, K, J, T = key
    per = runs[key]
    first = [per[li][min(per[li])] for li in sorted(per)]
    last = [per[li][max(per[li])] for li in sorted(per)]
    print("%-40s %6d %8d %10.4g %10.4g" % (
        f"K={K} J={J} T={T:g}", len(per), len(next(iter(per.values()))),
        float(np.median(first)), float(np.median(last))))
