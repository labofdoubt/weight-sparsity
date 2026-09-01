"""Validation-loss comparison of two run families, as a gradient-coloured plot.

Built for hard TopK swept over K against Top-(K+J) swept over J, but the two
families are given as name patterns so any pair of sweeps works:

    python scripts/plot_topk_vs_topkj_val_ce.py --runs-dir /workspace/runs \
        --out /workspace/plots/topk_vs_topkj_val_ce.png

Reads ``metrics.jsonl`` rather than the TensorBoard event files: it is the same
data, needs no TB dependency, and ``config.json`` sits beside it so the curves
can be checked against what actually ran instead of trusting the run name.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(run_dir):
    """``(steps, val_ce, config, finished)`` for one run, or None if absent."""
    metrics = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics):
        return None
    recs = [json.loads(l) for l in open(metrics)]
    pts = [(r["step"], r["val/ce"]) for r in recs if "val/ce" in r]
    if not pts:
        return None
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    finished = os.path.exists(os.path.join(run_dir, "summary.json"))
    return [s for s, _ in pts], [v for _, v in pts], cfg, finished


def family(ax, runs_dir, pattern, values, cmap, style, label_fmt, lo=0.35, hi=0.95):
    """Plot one sweep, shaded light->dark in the order the values are given."""
    shades = plt.get_cmap(cmap)(np.linspace(lo, hi, max(2, len(values))))
    plotted, missing = [], []
    for colour, value in zip(shades, values):
        name = pattern.format(value)
        got = load(os.path.join(runs_dir, name))
        if got is None:
            missing.append(name)
            continue
        steps, ce, cfg, finished = got
        b = cfg.get("activation_bottleneck", {})
        # trust the config over the run name
        shown = label_fmt.format(value)
        if b:
            shown = label_fmt.format(value) + f"  (k={b.get('k')}, j={b.get('j')})"
        if not finished:
            shown += "  [running]"
        ax.plot(steps, ce, style, color=colour, lw=1.9,
                label=shown, marker="" if finished else ".", ms=3)
        plotted.append((name, min(ce), finished))
    return plotted, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="/workspace/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard-pattern", default="res_hard_selcorr_k{}")
    ap.add_argument("--hard-values", default="64,128,256,512")
    ap.add_argument("--hard-label", default="hard TopK, K={}")
    ap.add_argument("--hard-cmap", default="Reds")
    ap.add_argument("--soft-pattern", default="res_soft_j{}_selcorr_k64")
    ap.add_argument("--soft-values", default="64,128,256")
    ap.add_argument("--soft-label", default="Top(K+J), K=64, J={}")
    ap.add_argument("--soft-cmap", default="Blues")
    ap.add_argument("--title", default="Residual-stream bottleneck: hard TopK vs Top-(K+J)")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--legend-size", type=float, default=12.0)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 6.8))
    hard, miss_h = family(
        ax, args.runs_dir, args.hard_pattern,
        [v.strip() for v in args.hard_values.split(",") if v.strip()],
        args.hard_cmap, "--", args.hard_label,
    )
    soft, miss_s = family(
        ax, args.runs_dir, args.soft_pattern,
        [v.strip() for v in args.soft_values.split(",") if v.strip()],
        args.soft_cmap, "-", args.soft_label,
    )

    ax.set_xlabel("step", fontsize=12)
    ax.set_ylabel("validation cross-entropy", fontsize=12)
    ax.set_title(args.title, fontsize=13)
    ax.tick_params(labelsize=11)
    if args.ymax:
        ax.set_ylim(top=args.ymax)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=args.legend_size, frameon=False, ncol=1, loc="upper right",
              title="dashed = hard TopK    solid = Top(K+J)",
              title_fontsize=args.legend_size + 1, labelspacing=0.5,
              handlelength=2.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"[plot] wrote {args.out}")
    for name, best, finished in sorted(hard + soft, key=lambda r: r[1]):
        print(f"  {name:30} best val {best:.4f}{'' if finished else '   (still running)'}")
    for m in miss_h + miss_s:
        print(f"  {m:30} MISSING -- skipped")


if __name__ == "__main__":
    main()
