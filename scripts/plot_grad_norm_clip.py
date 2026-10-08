"""Gradient norm against the clipping threshold, from a run's metrics.jsonl.

For each run: the logged pre-clip global gradient norm per step
(``train/grad_norm``) with the clip threshold (``train.grad_clip``), and the
boundary score ``b`` with the realized kernel width.  (The gate diagnostic
``rb_support_grad_norm`` is a per-token score-space norm of the support term
inside the gate, not a parameter-gradient norm, so it is not drawn here.)  Per 500-step
window the share of steps whose norm exceeded the threshold -- the clipping
budget the run consumed -- is printed and drawn as bars.

    python scripts/plot_grad_norm_clip.py --out docs/code-residual-analysis \
        /workspace/runs/<run> [...]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def read_run(run_dir: str):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    rows = [r for r in rows if "train/grad_norm" in r]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    clip = float(cfg["train"].get("grad_clip", 0.0))
    step = np.array([r["step"] for r in rows])
    get = lambda key: np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return dict(name=os.path.basename(run_dir.rstrip("/")), step=step, clip=clip,
                gnorm=get("train/grad_norm"), sup=get("bottleneck/rb_support_grad_norm"),
                b=get("bottleneck/rb_boundary"), t_eff=get("bottleneck/rb_temperature_eff"),
                scale_eff=get("bottleneck/rb_support_scale_eff"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories holding metrics.jsonl + config.json")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [read_run(r) for r in args.runs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for n, r in enumerate(runs):
        c = COLORS[n % len(COLORS)]
        axes[0].plot(r["step"], r["gnorm"], color=c, lw=0.8, alpha=0.8, label=r["name"])
        axes[1].plot(r["step"], r["b"], color=c, lw=0.9, label=f"{r['name']}  b")
        if np.isfinite(r["t_eff"]).any():
            axes[1].plot(r["step"], r["t_eff"], color=c, lw=0.9, ls="--", label=f"{r['name']}  T_eff")
        # clipping budget per window
        edges = np.arange(0, r["step"].max() + args.window, args.window)
        frac, mean = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r["step"] > lo) & (r["step"] <= hi)
            frac.append(float((r["gnorm"][m] > r["clip"]).mean()) if m.any() else np.nan)
            mean.append(float(r["gnorm"][m].mean()) if m.any() else np.nan)
        centers = (edges[:-1] + edges[1:]) / 2
        axes[2].bar(centers + (n - (len(runs) - 1) / 2) * args.window * 0.28, frac,
                    width=args.window * 0.28, color=c, alpha=0.8, label=r["name"])
        print(f"{r['name']}: clip {r['clip']}; per {args.window}-step window "
              + "  ".join(f"({int(lo)},{int(hi)}]: {f * 100:4.1f}% clipped, mean |g| {mu:.2f}"
                          for lo, hi, f, mu in zip(edges[:-1], edges[1:], frac, mean)))
    clip = runs[0]["clip"]
    axes[0].axhline(clip, color="k", lw=1, ls="--", label=f"clip threshold {clip:g}")
    axes[0].set_yscale("log"); axes[0].set_title("pre-clip global gradient norm per step")
    axes[0].set_xlabel("step"); axes[0].legend(fontsize=7)
    axes[1].set_yscale("log"); axes[1].set_title("boundary score b = s_(K+1) (and realized kernel width)")
    axes[1].set_xlabel("step"); axes[1].legend(fontsize=7)
    axes[2].set_ylim(0, 1); axes[2].set_title(f"share of steps with |grad| > clip, per {args.window} steps")
    axes[2].set_xlabel("step"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.out, "grad_norm_clip.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
