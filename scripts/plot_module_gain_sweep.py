"""Layer-averaged module gain vs K, from several analysis/module_gain.py runs.

    python scripts/plot_module_gain_sweep.py out.png gain_k32.json gain_k64.json ...

Four panels: gradient and forward amplification, for the bottleneck
(encoder through decoder) and for its post-RMSNorm.  Each point is the
GEOMETRIC mean over layers -- these gains compound multiplicatively down the
stack, so that is the average whose 24th power is the whole-stack factor -- with
the per-layer min/max as a band.  The dotted line at 1.0 is transparency.

The bottleneck's backward panel also carries ``K / d_model``, which is what the
measurement follows to better than 3%: the mask keeps K of N coordinates, so
the composite encoder-mask-decoder map has rank at most K against a stream of
dimension d_model.  Verified N-independent -- at K=512 the gain is 0.501,
0.500, 0.499 for N = 2048, 4096, 8192, where 4K/N would have given 1.0, 0.5,
0.25.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, ACCENT, REF = "#332288", "#CC3311", "#666666"
PANELS = [
    ("bwd_bottleneck", "gradient: bottleneck (encoder $\\to$ decoder)",
     r"$\langle\|dL/dx\|^2\rangle \,/\, \langle\|dL/dy\|^2\rangle$", INK, True),
    ("bwd_post_norm", "gradient: post-RMSNorm",
     r"$\langle\|dL/dy\|^2\rangle \,/\, \langle\|dL/dz\|^2\rangle$", INK, False),
    ("fwd_bottleneck", "forward: bottleneck (encoder $\\to$ decoder)",
     r"$\langle\|y\|^2\rangle \,/\, \langle\|x\|^2\rangle$", ACCENT, False),
    ("fwd_post_norm", "forward: post-RMSNorm",
     r"$\langle\|z\|^2\rangle \,/\, \langle\|y\|^2\rangle$", ACCENT, False),
]


def main() -> None:
    out_path, paths = sys.argv[1], sys.argv[2:]
    runs = sorted((json.load(open(p)) for p in paths), key=lambda d: d["k"])
    ks = [d["k"] for d in runs]
    n_features = {d["n_features"] for d in runs}
    assert len(n_features) == 1, f"mixed n_features: {n_features}"
    N = n_features.pop()
    d_models = {d["d_model"] for d in runs}
    assert len(d_models) == 1, f"mixed d_model: {d_models}"
    d_model = d_models.pop()

    alphas = {round(d.get("md_alpha", 1.0), 6) for d in runs}
    scaled = alphas != {1.0}
    gammas = {str(d.get("pnorm_gamma", "1.0")) for d in runs}
    regamma = gammas != {"1.0"}

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    for ax, (field, name, ylab, color, ref) in zip(axes.ravel(), PANELS):
        if any(d["summary"].get(field) is None for d in runs):
            # the module is absent in this configuration (post_norm: false),
            # so the panel stays, empty and labelled, to keep the 2x2 layout
            # comparable with the runs that have it
            ax.text(0.5, 0.5, "no post-norm\nin this configuration",
                    ha="center", va="center", fontsize=11, color=REF,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=11.5)
            continue
        mean = [d["summary"][field]["geometric_mean"] for d in runs]
        lo = [d["summary"][field]["min"] for d in runs]
        hi = [d["summary"][field]["max"] for d in runs]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18, lw=0,
                        label="per-layer min-max")
        ax.plot(ks, mean, "-o", ms=5, color=color, label="geometric mean over layers")
        if ref:
            lab = f"$K/d_{{model}}$  ($d={d_model}$)"
            if scaled:
                lab += ", before the gain spread"
            ax.plot(ks, [k / d_model for k in ks], ":", color=REF, lw=1.4,
                    label=lab)
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K (active features of N)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(name, fontsize=11.5)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8.5, loc="best")

    d0 = runs[0]
    tail = ""
    if scaled:
        tail += ("\n" + r"with $\alpha=\sqrt{d_{model}/K}$ on each bottleneck's "
                 r"output, spread equally over the 4 MD gain vectors of its two "
                 r"projections")
    if regamma:
        tail += (", and the post-norm's $\gamma$ at "
                 r"$1/\sqrt{G_{bwd}}$ per layer")
    fig.suptitle(
        f"layer-averaged amplification vs K -- {d0['n_layers']}L d{d0['d_model']}, "
        f"N={N}, {d0['placement']}, post-norm, at init" + tail,
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(out_path)

    print(f"{'K':>6} {'bwd_bn':>9} {'/(K/d)':>7} {'bwd_pn':>9} "
          f"{'fwd_bn':>9} {'fwd_pn':>9} {'fwd net':>8} {'bwd net':>8}")
    for d in runs:
        g = {f: (d["summary"][f] or {}).get("geometric_mean", float("nan"))
             for f, *_ in PANELS}
        print(f"{d['k']:6d} {g['bwd_bottleneck']:9.4f} "
              f"{g['bwd_bottleneck'] / (d['k'] / d_model):7.3f} {g['bwd_post_norm']:9.4f} "
              f"{g['fwd_bottleneck']:9.4f} {g['fwd_post_norm']:9.4f} "
              f"{g['fwd_bottleneck'] * g['fwd_post_norm']:8.4f} "
              f"{g['bwd_bottleneck'] * g['bwd_post_norm']:8.4f}")


if __name__ == "__main__":
    main()
