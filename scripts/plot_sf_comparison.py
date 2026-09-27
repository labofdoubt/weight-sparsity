"""rblapsum_sf vs rblapsum (kappa) validation curves, grouped by pool K' = K+J.

One figure per pool: left panel T=2, right panel post-norm.  Two gradient
families -- kappa in blues, sf in oranges, both light -> dark with increasing
K -- and hard Top-K' (post-norm) as a black dashed reference.  The sf curves
here are val/ce, the HARD Top-K eval forward, directly comparable with every
other curve; the soft forward the sf models actually train is shown in the
separate soft-eval figure (sf only, same hard references).

Usage: python plot_sf.py out_dir/ sfcmp_curves.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

outdir, src = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
runs = json.load(open(src))

POOLS = [64, 128, 256, 512]
KS = [32, 64, 128, 256]
# light -> dark with increasing K (positions fixed by K, not by presence)
BLUES = {32: "#9EC9E2", 64: "#5BA3CF", 128: "#2171B5", 256: "#0B3D91"}
ORANGES = {32: "#FDBE85", 64: "#FD8D3C", 128: "#D94801", 256: "#7F2704"}

def sel(mode, var, k, j):
    for n, r in runs.items():
        if r["mode"] != mode or r["k"] != k or r["j"] != j:
            continue
        if var == "t2" and (r["T"] == 2.0 and not r["pn"]):
            return n, r
        if var == "pnorm" and (r["T"] == 1.0 and r["pn"]):
            return n, r
    return None, None

def hard_ref(pool):
    for n, r in runs.items():
        if r["mode"] == "hard" and r["k"] == pool and r["pn"]:
            return n, r
    return None, None

def draw(ax, pool, var, metric="val"):
    hn, hr = hard_ref(pool)
    if hr:
        s, v = zip(*hr["val"])
        ax.plot(s, v, ls="--", color="black", lw=1.8, zorder=3)
    tops = []
    for k in KS:
        j = pool - k
        if j <= 0:
            continue
        for mode, ramp in (("rblapsum", BLUES), ("rblapsum_sf", ORANGES)):
            n, r = sel(mode, var, k, j)
            if not r:
                continue
            series = r[metric] if metric != "val" else r["val"]
            if not series:
                continue
            s, v = zip(*series)
            ax.plot(s, v, color=ramp[k], lw=1.9,
                    alpha=0.95 if mode == "rblapsum_sf" else 0.85)
            if s[-1] < 19000:  # guard-stopped
                ax.plot([s[-1]], [min(v[-1], ax.get_ylim()[1])], marker="x",
                        color=ramp[k], ms=10, mew=2.4, clip_on=False, zorder=6)
            tops.append(v[-1])
    ax.set_xlim(0, 20000)
    ax.grid(alpha=0.3)
    ax.set_xlabel("training step")
    return tops

for pool in POOLS:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    hi = 0.0
    for ax, var, ttl in ((axes[0], "t2", "$T=2$"),
                         (axes[1], "pnorm", "post-norm")):
        tops = draw(ax, pool, var)
        ax.set_title(ttl, fontsize=12)
        hi = max([hi] + [t for t in tops if np.isfinite(t)])
    ylo = 1.33
    yhi = min(2.85, max(1.75, hi + 0.08))
    for ax in axes:
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("validation CE (nats), hard Top-$K$ eval")
    handles = [Line2D([0], [0], ls="--", color="black", lw=1.8,
                      label=f"hard Top-{pool} (post-norm)")]
    for k in KS:
        if pool - k <= 0:
            continue
        handles.append(Line2D([0], [0], color=BLUES[k], lw=2.2,
                              label=f"kappa $K{{=}}{k}$, $J{{=}}{pool-k}$"))
        handles.append(Line2D([0], [0], color=ORANGES[k], lw=2.2,
                              label=f"sf $K{{=}}{k}$, $J{{=}}{pool-k}$"))
    axes[1].legend(handles=handles, fontsize=8.5, loc="upper right",
                   ncol=2, framealpha=0.93)
    fig.suptitle(f"Pool $K{{+}}J={pool}$: soft forward (sf) vs hard forward "
                 "(kappa), hard-eval CE", fontsize=13, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"sf_pool_k{pool}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)

# ---- the soft forward the sf models actually train ------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharex=True)
for ax, pool in zip(axes.ravel(), POOLS):
    hn, hr = hard_ref(pool)
    if hr:
        s, v = zip(*hr["val"])
        ax.plot(s, v, ls="--", color="black", lw=1.8)
    for var, ls in (("t2", "-"), ("pnorm", ":")):
        for k in KS:
            j = pool - k
            if j <= 0:
                continue
            n, r = sel("rblapsum_sf", var, k, j)
            if not r or not r["val_soft"]:
                continue
            s, v = zip(*r["val_soft"])
            ax.plot(s, v, ls=ls, color=ORANGES[k], lw=1.9)
    ax.set_ylim(1.30, 1.75)
    ax.grid(alpha=0.3)
    ax.set_title(f"pool $K{{+}}J={pool}$", fontsize=11)
h = [Line2D([0], [0], ls="--", color="black", lw=1.8, label="hard Top-$K'$ (post-norm), hard eval")]
for k in KS:
    h.append(Line2D([0], [0], color=ORANGES[k], lw=2.2, label=f"sf $K{{=}}{k}$"))
h += [Line2D([0], [0], color="0.4", ls="-", lw=2, label="sf $T{=}2$"),
      Line2D([0], [0], color="0.4", ls=":", lw=2, label="sf post-norm")]
axes[0][0].legend(handles=h, fontsize=8.5, loc="upper right", ncol=2)
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("validation CE (nats), soft eval")
fig.suptitle("The sf models under their own (soft) forward: val_soft/ce against "
             "the hard Top-$K'$ ladder", fontsize=13, y=0.997)
fig.tight_layout()
p = os.path.join(outdir, "sf_soft_eval.png")
fig.savefig(p, dpi=170, bbox_inches="tight")
print(p)
