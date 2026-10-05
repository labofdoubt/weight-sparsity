"""Figures for docs/carry-surrogate.tex (the carry-aware RBLapSum scopes).

Reads docs/figures/carry/data: analysis/surrogate_probe.py JSONs at
initialization (init/), analysis/code_runs.py JSONs (runs/), the validation
curves of the new screens (curves_bergen.json) and of the code-carried K+J
grid (../kj-cr/data/curves_new.json); writes docs/figures/carry/fig_*.{pdf,png}.

  python scripts/plot_carry.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "carry", "data")
KJ = os.path.join(ROOT, "docs", "figures", "kj-cr", "data")
OUT = os.path.join(ROOT, "docs", "figures", "carry")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GRID = "#e1e0d9"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.0,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 3.0, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# scope -> (label, color, linestyle)
STYLE = {
    "hard": ("hard Top-$K$", INK, "-"),
    "pool": ("RBLapSum, pool", ORANGE, "-"),
    "fo": ("first order", YELLOW, "-"),
    "cl": ("carry, local", BLUE, "-"),
    "cm": ("carry, mixed", AQUA, "-"),
    "cp": ("carry, persistent", MAGENTA, "--"),
    "ch": ("carry, hard", "#7a5cc6", "--"),
}


def load(*parts):
    p = os.path.join(*parts)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: the backward along the carried code at initialization
# --------------------------------------------------------------------------- #
def fig_init():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw={"wspace": 0.42})
    for ax, (kj, title) in zip(axes[:2], [("k32_j224", r"a  $K{=}32$, $J{=}224$, $T{=}2$"),
                                          ("k256_j256", r"b  $K{=}256$, $J{=}256$, $T{=}2$")]):
        hard = load(DATA, "init", f"init_{kj}_hard.json")
        h = [g["code_grad_rms"] for g in hard["gates"]]
        for key in ("pool", "fo", "ch", "cp", "cm", "cl"):
            d = load(DATA, "init", f"init_{kj}_{key}.json")
            if d is None:
                continue
            lab, color, ls = STYLE[key]
            y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
            ax.plot(range(len(y)), y, ls, color=color, lw=1.5 if key in ("pool", "cl", "cm") else 1.0,
                    label=lab)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(8))
        ax.set_xlabel(r"gate $\ell$ (0 = input side, 7 = last)")
        ax.set_ylabel(r"RMS $\partial\mathcal{L}/\partial c_{\ell+1}$ / hard")
        ax.set_title(title)
        if kj == "k32_j224":
            ax.legend(loc="upper right", fontsize=5.4)
    # (c) the kernel width: T = 1 versus T = 2 at K = 32
    ax = axes[2]
    hard = load(DATA, "init", "init_k32_j224_hard.json")
    h = [g["code_grad_rms"] for g in hard["gates"]]
    for key, suffix, ls in (("pool", "", "-"), ("pool", "_t1", ":"), ("cm", "", "-"), ("cm", "_t1", ":"),
                            ("cl", "", "-"), ("cl", "_t1", ":")):
        d = load(DATA, "init", f"init_k32_j224_{key}{suffix}.json")
        if d is None:
            continue
        lab, color, _ = STYLE[key]
        y = [g["code_grad_rms"] / hh for g, hh in zip(d["gates"], h)]
        ax.plot(range(len(y)), y, ls, color=color, lw=1.3,
                label=lab + (r", $T{=}1$" if suffix else r", $T{=}2$"))
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(8))
    ax.set_xlabel(r"gate $\ell$")
    ax.set_title(r"c  $K{=}32$: kernel width")
    ax.legend(loc="upper right", fontsize=5.2)
    save(fig, "fig_init")


def table_init():
    """Per-block growth factor and surrogate energy, for the note."""
    for kj in ("k32_j224", "k256_j256"):
        print(kj)
        for key in ("hard", "pool", "fo", "cl", "ch", "cp", "cm"):
            for suffix in ("", "_t1"):
                d = load(DATA, "init", f"init_{kj}_{key}{suffix}.json")
                if d is None:
                    continue
                hard = load(DATA, "init", f"init_{kj}_hard.json")
                g0 = d["gates"][0]["code_grad_rms"] / hard["gates"][0]["code_grad_rms"]
                sig = sum(g["sigma"] for g in d["gates"]) / len(d["gates"])
                print(f"  {key + suffix:10s} gate0/hard {g0:7.2f}  per block x{d.get('per_block_factor', float('nan')):.3f}"
                      f"  sigma {sig:.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_init()
    table_init()
