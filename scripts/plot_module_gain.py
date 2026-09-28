"""Plot analysis/module_gain.py output: two panels per direction, vs layer.

    python scripts/plot_module_gain.py gain.json out_prefix ["title"]

writes <out_prefix>_bwd.png (gradient gain) and <out_prefix>_fwd.png
(activation gain).  Each figure has one panel for the bottleneck (encoder
through decoder) and one for its post-norm.  Gains are energy ratios along the
direction of propagation, so 1.0 (the dashed line) is transparent, above it is
amplification and below it attenuation; the log axis keeps the two comparable.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACCENT = "#332288", "#CC3311"
PANELS = {
    "bwd": [("bwd_bottleneck", "bottleneck (encoder $\\to$ decoder)",
             r"$\langle\|dL/dx\|^2\rangle \,/\, \langle\|dL/dy\|^2\rangle$"),
            ("bwd_post_norm", "post-RMSNorm",
             r"$\langle\|dL/dy\|^2\rangle \,/\, \langle\|dL/dz\|^2\rangle$")],
    "fwd": [("fwd_bottleneck", "bottleneck (encoder $\\to$ decoder)",
             r"$\langle\|y\|^2\rangle \,/\, \langle\|x\|^2\rangle$"),
            ("fwd_post_norm", "post-RMSNorm",
             r"$\langle\|z\|^2\rangle \,/\, \langle\|y\|^2\rangle$")],
}
WHAT = {"bwd": "gradient amplification", "fwd": "forward amplification"}


def draw(d, direction, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for ax, (field, name, ylab) in zip(axes, PANELS[direction]):
        rows = [r for r in d["layers"] if field in r]
        li = [r["layer"] for r in rows]
        g = [r[field]["gain"] for r in rows]
        med = [r[field]["gain_median_token"] for r in rows]
        ax.plot(li, g, "-o", ms=4.5, color=INK, label="energy ratio")
        ax.plot(li, med, "--s", ms=3.5, color=ACCENT, alpha=0.85,
                label="median over tokens")
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("layer", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        s = d["summary"].get(field) or {}
        sub = (f"geo-mean {s['geometric_mean']:.3g} per layer, "
               f"$\\prod$ over {s['n']} = {s['product']:.3g}") if s else ""
        ax.set_title(f"{name}\n{sub}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"{WHAT[direction]}: {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(path)


def main() -> None:
    d = json.load(open(sys.argv[1]))
    prefix = sys.argv[2]
    head = sys.argv[3] if len(sys.argv) > 3 else (
        f"{d['n_layers']}L d{d['d_model']}, N={d['n_features']} K={d['k']}, "
        f"{d['placement']}" + (", post-norm" if d["post_norm"] else ""))
    title = f"{head} @ step {d['step']} (batch CE {d['ce']:.3f})"
    for direction in ("bwd", "fwd"):
        draw(d, direction, f"{prefix}_{direction}.png", title)


if __name__ == "__main__":
    main()
