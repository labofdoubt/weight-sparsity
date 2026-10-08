"""Two panels: validation CE with and without rblapsum_radial_project.

One panel per cell (K = 32 and K = 256), the run without the projection
against the run with it: validation CE at every validation step (markers) and
the training CE, smoothed over a window, as a faint line.  The boundary score
of each run is printed per validation step.

    python scripts/plot_radial_project.py --out docs/code-residual-analysis \
        --k32 /workspace/runs/<base> /workspace/runs/<proj> \
        --k256 /workspace/runs/<base> /workspace/runs/<proj>
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"base": "#2a78d6", "proj": "#eb6834"}


def read_run(run_dir: str):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    ab = cfg["activation_bottleneck"]
    tr = [(r["step"], r["train/ce"]) for r in rows if "train/ce" in r]
    va = [(r["step"], r["val/ce"]) for r in rows if "val/ce" in r]
    b = [(r["step"], r["bottleneck/rb_boundary"]) for r in rows if "bottleneck/rb_boundary" in r]
    return dict(name=os.path.basename(run_dir.rstrip("/")), k=ab["k"], j=ab["j"],
                proj=bool(ab.get("rblapsum_radial_project", False)),
                tau=ab["temperature"], s=ab.get("rblapsum_support_strength"),
                train=np.array(tr), val=np.array(va), b=np.array(b))


def smooth(y, w):
    if len(y) < w:
        return y
    c = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def panel(ax, runs, window):
    for r in runs:
        tag = "proj" if r["proj"] else "base"
        lab = ("with radial projection" if r["proj"] else "without (reference)")
        if len(r["train"]):
            ax.plot(r["train"][:, 0], smooth(r["train"][:, 1], window), color=COLORS[tag],
                    lw=0.8, alpha=0.35)
        if len(r["val"]):
            ax.plot(r["val"][:, 0], r["val"][:, 1], "-o", color=COLORS[tag], ms=4, lw=1.4,
                    label=f"{lab}: val CE {r['val'][-1, 1]:.4f} at step {int(r['val'][-1, 0])}")
    r0 = runs[0]
    ax.set_title(f"K = {r0['k']}, J = {r0['j']}  ·  first-order, span rule tau = {r0['tau']}, "
                 f"s = {r0['s']}", fontsize=10)
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k32", nargs=2, required=True, metavar=("BASE", "PROJ"))
    ap.add_argument("--k256", nargs=2, required=True, metavar=("BASE", "PROJ"))
    ap.add_argument("--window", type=int, default=10, help="training-CE smoothing, in log rows")
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, dirs in zip(axes, (args.k32, args.k256)):
        runs = [read_run(d) for d in dirs]
        panel(ax, runs, args.window)
        if args.ymax:
            lo = min(r["val"][:, 1].min() for r in runs if len(r["val"]))
            ax.set_ylim(lo - 0.05, args.ymax)
        for r in runs:
            steps = r["val"][:, 0] if len(r["val"]) else []
            bs = {int(s): float(r["b"][np.argmin(np.abs(r["b"][:, 0] - s)), 1]) for s in steps} if len(r["b"]) else {}
            print(f"{r['name']}: val CE " + ", ".join(f"{int(s)}: {v:.4f}" for s, v in r["val"])
                  + "  |  b " + ", ".join(f"{s}: {v:.2f}" for s, v in bs.items()))
    axes[0].set_ylabel("validation CE (markers), training CE smoothed (faint)")
    fig.tight_layout()
    out = os.path.join(args.out, "radial_project_ce.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
