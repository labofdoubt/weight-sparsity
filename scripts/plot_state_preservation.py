"""Per-token reconstruction error histograms (analysis/state_preservation.py).

One panel with the ridge map's per-token normalized test error for each
model given, one histogram per model with its mean (the reported epsilon)
in the legend, and a second panel with the identity map's error (c~ read as
a prediction of c, no fitting) for scale.

    python scripts/plot_state_preservation.py --out docs/code-residual-analysis \
        --label "hard Top-32" /workspace/analysis/state_pres/<run>_s2_e6.npz \
        --label "RBLapSum (32,224)" /workspace/analysis/state_pres/<run>_s2_e6.npz
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#e34948", "#2a78d6", "#1baf7a", "#eda100"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--label", action="append", default=[], help="one per npz, in order")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--name", default="state_preservation.png")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = []
    for i, p in enumerate(args.npz):
        z = np.load(p)
        m = json.load(open(os.path.splitext(p)[0] + ".json"))
        lab = args.label[i] if i < len(args.label) else m["run"]
        runs.append((lab, z, m))
    s, e = runs[0][2]["start"], runs[0][2]["end"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for which, ax, title in ((("err_ridge", "ridge"), axes[0], "ridge map R(c~) = A c~ + b, lambda chosen on validation"),
                             (("err_identity", "identity"), axes[1], "no map: c~ itself as the prediction of c")):
        key, ekey = which
        hi = args.xmax or max(np.percentile(z[key], 99) for _, z, _ in runs)
        bins = np.linspace(0, hi, args.bins + 1)
        for n, (lab, z, m) in enumerate(runs):
            v = np.clip(z[key], 0, hi)
            eps = m["errors"]["test"][ekey]
            ax.hist(v, bins=bins, color=COLORS[n % len(COLORS)], alpha=0.5, edgecolor="none",
                    label=f"{lab}: eps = {eps:.3f} (median {np.median(z[key]):.3f})")
            ax.axvline(eps, color=COLORS[n % len(COLORS)], lw=1.2, ls="--")
        ax.axvline(1.0, color="k", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10); ax.set_xlabel("per-token ||prediction - c||^2 / E||c - E[c]||^2")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylabel("test tokens")
    fig.suptitle(f"State preservation, bottleneck {s} -> {e} with block updates off: "
                 f"reconstruction of c_{s} from the transported c_{e}  "
                 f"({runs[0][2]['tokens']['test']:,} test tokens; dashed = mean, dotted = constant predictor)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = os.path.join(args.out, args.name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
