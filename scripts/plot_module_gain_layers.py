"""Per-layer gradient amplification vs K, from analysis/module_gain.py runs.

    python scripts/plot_module_gain_layers.py out_prefix gain_k32.json ... [--field bwd_bottleneck]

writes two views of the same numbers:

  <out_prefix>_overlay.png   all layers on one axes, coloured by depth
  <out_prefix>_grid.png      one small panel per layer, shared axes

The layer average hides what these show: a curve can sit at one value in the
shallow layers and another in the deep ones, which is exactly what happens once
the forward gain is not 1 and the stream leaves its initial scale. The dotted
line at 1.0 is transparency; the grey dashed line is the same run's *forward*
gain (layer-averaged), the other value the per-layer curves are drawn to.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

REF = "#666666"
YLAB = {
    "bwd_bottleneck": r"$\langle\|dL/dx\|^2\rangle / \langle\|dL/dy\|^2\rangle$",
    "bwd_post_norm": r"$\langle\|dL/dy\|^2\rangle / \langle\|dL/dz\|^2\rangle$",
    "fwd_bottleneck": r"$\langle\|y\|^2\rangle / \langle\|x\|^2\rangle$",
    "fwd_post_norm": r"$\langle\|z\|^2\rangle / \langle\|y\|^2\rangle$",
}
NAME = {
    "bwd_bottleneck": "gradient amplification, bottleneck (encoder to decoder)",
    "bwd_post_norm": "gradient amplification, post-RMSNorm",
    "fwd_bottleneck": "forward amplification, bottleneck (encoder to decoder)",
    "fwd_post_norm": "forward amplification, post-RMSNorm",
}


def collect(paths, field):
    """``(ks, {layer: [gain per K]}, fwd_mean_per_K, meta)`` over sorted runs."""
    runs = sorted((json.load(open(p)) for p in paths), key=lambda d: d["k"])
    ks = [d["k"] for d in runs]
    layers = sorted({r["layer"] for d in runs for r in d["layers"] if field in r})
    series = {}
    for li in layers:
        vals = []
        for d in runs:
            rec = next((r for r in d["layers"] if r["layer"] == li), {})
            vals.append(rec.get(field, {}).get("gain", float("nan")))
        series[li] = vals
    twin = "fwd_bottleneck" if field.startswith("bwd") else "bwd_bottleneck"
    ref = {"mean": [(d["summary"].get(twin) or {}).get("geometric_mean", float("nan"))
                    for d in runs]}
    for li in layers:
        vals = []
        for d in runs:
            rec = next((r for r in d["layers"] if r["layer"] == li), {})
            vals.append(rec.get(twin, {}).get("gain", float("nan")))
        ref[li] = vals
    return ks, series, ref, runs[0]


def title(meta, field):
    alpha = round(meta.get("md_alpha", 1.0), 6) != 1.0
    head = (f"{NAME[field]} per layer -- {meta['n_layers']}L d{meta['d_model']}, "
            f"N={meta['n_features']}, {meta['placement']}, "
            + ("post-norm" if meta.get("post_norm", True) else "no post-norm")
            + ", at init")
    if alpha:
        head += ("\n" + r"with $\alpha=\sqrt{d_{model}/K}$ on each bottleneck's "
                 r"output, spread over its 4 MD gain vectors")
    return head


def style(ax, ks, field, legend=True, small=False):
    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7 if small else 10)
    ax.grid(alpha=0.3, which="both")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    if not small:
        ax.set_xlabel("K (active features of N)", fontsize=11)
        ax.set_ylabel(YLAB[field], fontsize=11)


def overlay(ks, series, ref, meta, field, path):
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    norm = colors.Normalize(vmin=min(series), vmax=max(series))
    cmap = cm.viridis
    for li, vals in series.items():
        ax.plot(ks, vals, "-o", ms=3.5, lw=1.4, color=cmap(norm(li)), alpha=0.9)
    ax.plot(ks, ref["mean"], "--", color=REF, lw=1.6,
            label=f"{'forward' if field.startswith('bwd') else 'gradient'} "
                  "gain of the same runs (layer mean)")
    style(ax, ks, field)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="layer")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title(meta, field), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(path)


def grid(ks, series, ref, meta, field, path, ncols=4):
    lis = sorted(series)
    nrows = -(-len(lis) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.15 * nrows),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    lo = min(v for vals in series.values() for v in vals if v == v)
    hi = max(v for vals in series.values() for v in vals if v == v)
    for ax, li in zip(flat, lis):
        ax.plot(ks, series[li], "-o", ms=3, lw=1.3, color="#332288")
        ax.plot(ks, ref[li], "--", color=REF, lw=1.1)
        style(ax, ks, field, small=True)
        ax.set_ylim(lo * 0.7, hi * 1.4)
        ax.set_title(f"layer {li}", fontsize=9.5)
        ax.tick_params(labelsize=7)
    for ax in flat[len(lis):]:
        ax.axis("off")
    fig.supxlabel("K (active features of N)", fontsize=11)
    fig.supylabel(YLAB[field], fontsize=11)
    twin_name = "forward" if field.startswith("bwd") else "gradient"
    fig.suptitle(title(meta, field)
                 + f"   (dashed: that layer's {twin_name} gain)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(path)


def main() -> None:
    args = sys.argv[1:]
    field = "bwd_bottleneck"
    if "--field" in args:
        i = args.index("--field")
        field = args[i + 1]
        del args[i:i + 2]
    prefix, paths = args[0], args[1:]
    ks, series, ref, meta = collect(paths, field)
    assert series, f"no layer carries {field}"
    overlay(ks, series, ref, meta, field, f"{prefix}_overlay.png")
    grid(ks, series, ref, meta, field, f"{prefix}_grid.png")


if __name__ == "__main__":
    main()
