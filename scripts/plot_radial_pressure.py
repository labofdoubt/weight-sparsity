"""Figures for the radial-pressure probe (analysis/radial_pressure_probe.py).

For each run: a grid with one row per quantity (radial pressure p, radial
norm fraction f, cosine c of the radial surrogate component with the hard
gradient) and one column per checkpoint step, every panel holding the
token-wise distribution for a few bottlenecks.  Plus one summary figure over
all runs: per-layer medians of p and f against the step, with the mean
boundary score b on a second axis -- the drift the pressure is supposed to
predict.

    python scripts/plot_radial_pressure.py --out docs/code-residual-analysis \
        /workspace/analysis/radial/<run>.npz [...]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAYER_COLORS = {1: "#2a78d6", 3: "#eb6834", 5: "#1baf7a", 0: "#52514e", 2: "#eda100",
                4: "#e87ba4", 6: "#4a3aa7", 7: "#e34948"}
QUANT = (("p", "radial pressure  <s̄, g_sur> / ||s̄||²", "symlog"),
         ("f", "radial norm fraction  <s̄, g_sur>² / (||s̄||² ||g_sur||²)", "linear"),
         ("c", "cos(radial surrogate component, hard gradient)", "linear"))


def load(path: str):
    z = np.load(path)
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    steps = sorted(int(s) for s in meta["steps"])
    return z, meta, steps


def per_run_figure(path: str, layers, out_dir: str) -> str:
    z, meta, steps = load(path)
    run = meta["run"]
    fig, axes = plt.subplots(len(QUANT), len(steps), figsize=(3.6 * len(steps), 9.2),
                             sharey="row", squeeze=False)
    for col, step in enumerate(steps):
        for row, (q, label, scale) in enumerate(QUANT):
            ax = axes[row][col]
            vals = {li: z[f"{step}/{li}/{q}"] for li in layers}
            if q == "p":
                allv = np.concatenate(list(vals.values()))
                lim = np.percentile(np.abs(allv), 99.5)
                bins = np.linspace(-lim, lim, 61)
            elif q == "f":
                bins = np.linspace(0, 1, 51)
            else:
                bins = np.linspace(-1, 1, 51)
            for li, v in vals.items():
                v = np.clip(v, bins[0], bins[-1])
                ax.hist(v, bins=bins, color=LAYER_COLORS.get(li, "#888"), alpha=0.5,
                        label=f"block {li}  (median {np.median(vals[li]):+.3g})", edgecolor="none")
            if q == "p":
                ax.axvline(0, color="k", lw=0.8, ls=":")
            if q == "c":
                ax.axvline(0, color="k", lw=0.8, ls=":")
            ax.legend(fontsize=7, loc="upper right")
            if row == 0:
                b = ", ".join(f"{meta['steps'][str(step)]['layers'][str(li)]['b_mean']:.2f}" for li in layers)
                gn = np.mean(meta["steps"][str(step)]["grad_norm_total"])
                gh = np.mean(meta["steps"][str(step)]["grad_norm_hard_only"])
                ax.set_title(f"step {step}\nb = {b}\n|grad| {gn:.2f}, hard-only {gh:.2f}", fontsize=8)
            if row == len(QUANT) - 1:
                ax.set_xlabel(label, fontsize=8)
            elif col == 0:
                ax.set_xlabel(label, fontsize=8)
        axes[0][col].set_ylabel("tokens" if col == 0 else "")
    sup = (f"{run}: K={meta['k']} J={meta['j']} scope={meta['scope']} width={meta['kernel_width']} "
           f"tau={meta['temperature']} strength={meta['support_strength']}  ·  "
           f"{meta['batch']} x {meta['seq_len']} tokens x {len(meta['offsets'])} training batches")
    fig.suptitle(sup, fontsize=10, y=1.0)
    fig.tight_layout()
    out = os.path.join(out_dir, f"radial_{run}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def summary_figure(paths, layers, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    styles = ["-", "--", ":", "-."]
    for n, path in enumerate(paths):
        z, meta, steps = load(path)
        run = meta["run"]
        for li in layers:
            col = LAYER_COLORS.get(li, "#888")
            med_p = [meta["steps"][str(s)]["layers"][str(li)]["p_q"][2] for s in steps]
            med_f = [meta["steps"][str(s)]["layers"][str(li)]["f_q"][2] for s in steps]
            b = [meta["steps"][str(s)]["layers"][str(li)]["b_mean"] for s in steps]
            lab = f"{run}  block {li}"
            axes[0].plot(steps, med_p, styles[n % 4], color=col, marker="o", ms=3, label=lab)
            axes[1].plot(steps, med_f, styles[n % 4], color=col, marker="o", ms=3, label=lab)
            axes[2].plot(steps, b, styles[n % 4], color=col, marker="o", ms=3, label=lab)
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_title("median radial pressure p"); axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_title("median radial norm fraction f"); axes[1].set_ylim(0, 1)
    axes[2].set_title("mean boundary score b = s_(K+1)"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[2].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    out = os.path.join(out_dir, "radial_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--layers", default="1,3,5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)
    for p in args.npz:
        print(per_run_figure(p, layers, args.out))
    print(summary_figure(args.npz, layers, args.out))


if __name__ == "__main__":
    main()
