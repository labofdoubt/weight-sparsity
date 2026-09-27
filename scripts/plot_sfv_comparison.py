"""rblapsum_sf value_grad=support (vsup) vs rblapsum kappa, by pool.

Same layout as the sf note's figures: per pool, left T=2 / right post-norm,
kappa blues, the family under test oranges (light->dark with K), hard Top-K'
post-norm dashed black.  x-axis to 10k (the campaign's horizon on the 20k
schedule).  Data: vsup runs from vsup_curves.json; kappa and hard references
from sfcmp_curves.json truncated to the same horizon.

Usage: python plot_sfv.py out_dir/ sfcmp_curves.json vsup_curves.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

outdir, old_p, new_p = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
old = json.load(open(old_p))
new = json.load(open(new_p))
XMAX = 10000

def trunc(series):
    return [(s, v) for s, v in series if s <= XMAX]

POOLS = [64, 128, 256, 512]
KS = [32, 64, 128, 256]
BLUES = {32: "#9EC9E2", 64: "#5BA3CF", 128: "#2171B5", 256: "#0B3D91"}
ORANGES = {32: "#FDBE85", 64: "#FD8D3C", 128: "#D94801", 256: "#7F2704"}

def var_sel(var, r):
    return (r["T"] == 2.0 and not r["pn"]) if var == "t2" else (r["T"] == 1.0 and r["pn"])

def kappa_run(var, k, j):
    for r in old.values():
        if r["mode"] == "rblapsum" and r["k"] == k and r["j"] == j and var_sel(var, r):
            return r

def vsup_run(var, k, j):
    for r in new.values():
        if r.get("vg") == "support" and r["k"] == k and r["j"] == j and var_sel(var, r):
            return r

def hard_ref(pool):
    for r in old.values():
        if r["mode"] == "hard" and r["k"] == pool and r["pn"]:
            return r

for pool in POOLS:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    hi = 0.0
    for ax, var, ttl in ((axes[0], "t2", "$T=2$"), (axes[1], "pnorm", "post-norm")):
        hr = hard_ref(pool)
        if hr:
            s, v = zip(*trunc(hr["val"]))
            ax.plot(s, v, ls="--", color="black", lw=1.8, zorder=3)
        for k in KS:
            j = pool - k
            if j <= 0:
                continue
            for run, ramp in ((kappa_run(var, k, j), BLUES), (vsup_run(var, k, j), ORANGES)):
                if not run:
                    continue
                sv = trunc(run["val"])
                if not sv:
                    continue
                s, v = zip(*sv)
                ax.plot(s, v, color=ramp[k], lw=1.9,
                        alpha=0.95 if ramp is ORANGES else 0.85)
                if s[-1] < 9000:  # guard-stopped
                    ax.plot([s[-1]], [min(v[-1], 2.78)], marker="x", color=ramp[k],
                            ms=10, mew=2.4, clip_on=False, zorder=6)
                hi = max(hi, v[-1] if np.isfinite(v[-1]) else 0)
        ax.set_xlim(0, XMAX)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
        ax.set_title(ttl, fontsize=12)
    for ax in axes:
        ax.set_ylim(1.45, min(2.85, max(1.9, hi + 0.08)))
    axes[0].set_ylabel("validation CE (nats), hard Top-$K$ eval")
    handles = [Line2D([0], [0], ls="--", color="black", lw=1.8,
                      label=f"hard Top-{pool} (post-norm)")]
    for k in KS:
        if pool - k <= 0:
            continue
        handles.append(Line2D([0], [0], color=BLUES[k], lw=2.2,
                              label=f"kappa $K{{=}}{k}$, $J{{=}}{pool-k}$"))
        handles.append(Line2D([0], [0], color=ORANGES[k], lw=2.2,
                              label=f"sf-vsup $K{{=}}{k}$, $J{{=}}{pool-k}$"))
    axes[1].legend(handles=handles, fontsize=8.5, loc="upper right", ncol=2,
                   framealpha=0.93)
    fig.suptitle(f"Pool $K{{+}}J={pool}$: soft forward with support-only value "
                 "gradient (sf-vsup) vs hard forward (kappa), hard-eval CE",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"sfv_pool_k{pool}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print(p)

# soft-eval figure: vsup under its own forward vs the hard ladder
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharex=True)
for ax, pool in zip(axes.ravel(), POOLS):
    hr = hard_ref(pool)
    if hr:
        s, v = zip(*trunc(hr["val"]))
        ax.plot(s, v, ls="--", color="black", lw=1.8)
    for var, ls in (("t2", "-"), ("pnorm", ":")):
        for k in KS:
            j = pool - k
            if j <= 0:
                continue
            run = vsup_run(var, k, j)
            if not run or not run["val_soft"]:
                continue
            sv = trunc(run["val_soft"])
            s, v = zip(*sv)
            ax.plot(s, v, ls=ls, color=ORANGES[k], lw=1.9)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(1.45, 2.0)
    ax.grid(alpha=0.3)
    ax.set_title(f"pool $K{{+}}J={pool}$", fontsize=11)
h = [Line2D([0], [0], ls="--", color="black", lw=1.8,
            label="hard Top-$K'$ (post-norm), hard eval")]
for k in KS:
    h.append(Line2D([0], [0], color=ORANGES[k], lw=2.2, label=f"sf-vsup $K{{=}}{k}$"))
h += [Line2D([0], [0], color="0.4", ls="-", lw=2, label="sf-vsup $T{=}2$"),
      Line2D([0], [0], color="0.4", ls=":", lw=2, label="sf-vsup post-norm")]
axes[0][0].legend(handles=h, fontsize=8.5, loc="upper right", ncol=2)
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("validation CE (nats), soft eval")
fig.suptitle("sf-vsup under its own (soft) forward: val_soft/ce against the "
             "hard Top-$K'$ ladder", fontsize=13, y=0.997)
fig.tight_layout()
p = os.path.join(outdir, "sfv_soft_eval.png")
fig.savefig(p, dpi=170, bbox_inches="tight")
print(p)
