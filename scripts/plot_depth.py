"""Figures for docs/stream-bottleneck-depth*.tex (24-layer stream-bottleneck study).

Reads docs/figures/depth/data (analysis/depth_probe.py JSONs at init and at
step 2000, the curve extract) and the existing K sweep in
docs/figures/diag500m/module_gain, and writes docs/figures/depth/fig_*.pdf/png.

  python scripts/plot_depth.py
"""

from __future__ import annotations

import glob
import json
import math
import os
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "depth", "data")
OUT = os.path.join(ROOT, "docs", "figures", "depth")
KSWEEP = os.path.join(ROOT, "docs", "figures", "diag500m", "module_gain", "data")
N_FEATURES = 4096
UNIGRAM = 5.9665  # token-frequency entropy of TinyStories train.bin, nats

# reference palette in its fixed slot order; the failed configuration is the
# muted ink (emphasis), references are neutral ink
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.3,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 3.0,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def s2(rho: float) -> float:
    """Selection energy gain E[Z^2 | |Z| > t], P(|Z| > t) = rho (1 for rho >= 1)."""
    if rho >= 1:
        return 1.0
    t = NormalDist().inv_cdf(1 - rho / 2)
    phi = math.exp(-t * t / 2) / math.sqrt(2 * math.pi)
    return 1 + 2 * t * phi / rho


def load(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def per_layer(ax, d, key, color, marker=None, lw=1.4, norm_last=False, label=None):
    if d is None:
        return
    y = [r[key] for r in d["layers"]]
    if norm_last:
        y = [v / y[-1] for v in y]
    ax.plot(range(len(y)), y, "-", color=color, lw=lw, marker=marker, ms=2.8,
            mfc="white", mew=0.9, markevery=4, label=label)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: the two mechanisms, at initialization
# --------------------------------------------------------------------------- #
def fig_init():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"wspace": 0.36})

    # (a) selection gain: theory vs the K sweep of the 24-layer model
    ax = axes[0]
    rhos = [10 ** (x / 50) for x in range(-125, 1)]
    ax.plot(rhos, [s2(r) for r in rhos], color=INK, lw=1.0, label=r"$s^2(K/N)$, eq. (2)")
    pn, bp = [], []
    for f in glob.glob(os.path.join(KSWEEP, "k_sweep", "*.json")):
        j = json.load(open(f))
        sm = j["summary"]
        pn.append((j["k"] / j["n_features"], sm["fwd_bottleneck"]["geometric_mean"]
                   / sm["bwd_bottleneck"]["geometric_mean"]))
    for f in glob.glob(os.path.join(KSWEEP, "k_sweep_nopnorm_ortho_bp", "*.json")):
        j = json.load(open(f))
        bp.append((j["k"] / j["n_features"], j["summary"]["fwd_bottleneck"]["geometric_mean"]))
    pn.sort(); bp.sort()
    ax.plot(*zip(*pn), "o", ms=4.4, color=BLUE, mec="white", mew=0.8,
            label=r"$G_f/G_b$, post-norm")
    ax.plot(*zip(*bp), "s", ms=3.3, color=ORANGE, mec="white", mew=0.6,
            label=r"$G_f$ at $G_b=1$ (bp, orthogonal)")
    ax.set_xscale("log")
    ax.set_xlim(4e-3, 1.25)
    ax.set_ylim(0, 10)
    ax.set_xlabel(r"kept fraction $K/N$")
    ax.set_ylabel("energy gain of one bottleneck")
    ax.annotate(r"$4.02$ at $K/N=1/8$", (1 / 8, s2(1 / 8)), xytext=(0.30, 6.2),
                fontsize=6.2, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=3))
    ax.legend(loc="lower left")
    ax.set_title("a  forward/backward mismatch")

    # (b) gradient reaching block l at initialization (24 layers)
    ax = axes[1]
    L = 24
    ax.plot(range(L), [math.sqrt(s2(1 / 8)) ** (-(L - 1 - l)) for l in range(L)],
            color=INK, lw=0.7, label=r"$s^{-(23-\ell)}$")
    ax.plot(range(L), [math.sqrt(s2(32 / N_FEATURES)) ** (-(L - 1 - l)) for l in range(L)],
            color=INK, lw=0.5, alpha=0.6)
    per_layer(ax, load("init3", "k32_pnorm.json"), "g_fc1", LIGHT, norm_last=True,
              lw=1.1, label=r"$K{=}32$, post-norm")
    per_layer(ax, load("init3", "base_pnorm.json"), "g_fc1", MUTED, norm_last=True,
              label=r"$K{=}512$, post-norm")
    per_layer(ax, load("init3", "shift_pnorm.json"), "g_fc1", ORANGE, marker="o",
              norm_last=True, label="shift, post-norm")
    per_layer(ax, load("init3", "coderes_a1.json"), "g_fc1", BLUE, norm_last=True,
              label="code residual")
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 30)
    ax.set_xticks([0, 8, 16, 23])
    ax.set_xlabel(r"block $\ell$")
    ax.set_ylabel(r"$\|\nabla W_{\mathrm{fc1}}\|$ / last block")
    ax.legend(loc="lower right")
    ax.set_title("b  gradient at initialization")

    # (c) common-mode fraction of the stream: post-norm fixed points vs no norm
    ax = axes[2]
    pn_runs = [("kN_pnorm", N_FEATURES, r"$K{=}N$", 1.18),
               ("k2048_pnorm", 2048, r"$K{=}2048$", 0.80),
               ("base_pnorm", 512, r"$K{=}512$", 1.0),
               ("k32_pnorm", 32, r"$K{=}32$", 1.0)]
    for name, k, lab, nudge in pn_runs:
        d = load("init3", f"{name}.json")
        if d is None:
            continue
        per_layer(ax, d, "f_stream", MUTED, lw=1.2)
        sel = d["layers"][6:]
        rc = sum(r["r_common"] for r in sel) / len(sel)
        r = sum(r["r_delta"] for r in sel) / len(sel)
        kap = (k / N_FEATURES) * s2(k / N_FEATURES)
        fstar = kap * rc / (1 + r - kap)
        ax.plot([24.3], [fstar], "<", ms=4.0, color=INK, mec="white", mew=0.5)
        ax.text(25.2, fstar * nudge, lab, fontsize=6.0, color=INK2, va="center")
    ax.text(25.0, 2.1, r"$f^\ast$, eq. (4)", fontsize=6.0, color=INK2, va="center")
    for name, lab, color in (("kN_nopnorm", r"$K{=}N$, no norm", AQUA),
                             ("coderes_a1", "code residual, no norm", BLUE)):
        per_layer(ax, load("init3", f"{name}.json"), "f_stream", color, lw=1.3, label=lab)
    ax.plot([], [], "-", color=MUTED, lw=1.2, label="TopK + post-norm")
    ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0.0))
    ax.set_yscale("log")
    ax.set_ylim(1.2e-3, 3.0)
    ax.set_xlim(-0.5, 29)
    ax.set_xticks([0, 8, 16, 23])
    ax.set_xlabel(r"block $\ell$")
    ax.set_ylabel(r"common-mode fraction $f$ of the stream")
    ax.set_title("c  token-independent share at init")
    save(fig, "fig_init")


# --------------------------------------------------------------------------- #
# Figure 2: training
# --------------------------------------------------------------------------- #
FAILED_PN = ["li_24L_k512_pnorm_shift_energy", "li_24L_k512_pnorm_shift_fixed",
             "li_24L_k512_pnorm_shift_energy_shared", "li_24L_k512_pnorm_rblapsum_t1",
             "li_24L_k32_pnorm_shift_energy", "po_500m_hard_k512_pnorm_bp_std_shared",
             "li_24L_kN_pnorm", "al_500m_hard_k32_pnorm_md"]


def fig_training():
    curves = load("curves.json") or {}
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.95),
                             gridspec_kw={"wspace": 0.45, "width_ratios": [1.55, 1, 1, 0.9]})

    # (a) validation CE
    ax = axes[0]
    for r in FAILED_PN:
        c = curves.get(r)
        if c and c["val"]:
            s, v = zip(*c["val"])
            ax.plot(s, v, color=LIGHT, lw=0.9)
    named = [
        ("al_500m_hard_k512_pnorm_bp_std", "TopK, post-norm", MUTED, 1.4),
        ("al_8L_hard_k512_nopnorm_bp_std", "TopK, no norm, 8 L", AQUA, 1.4),
        ("li_24L_dense", "dense, no bottleneck", INK2, 1.0),
        ("li_24L_k512_nopnorm_shift_energy", "value shift, no norm", ORANGE, 1.4),
        ("li_24L_k512_coderes_a1", "code residual", BLUE, 1.6),
    ]
    for r, label, color, lw in named:
        c = curves.get(r)
        if not c or not c["val"]:
            continue
        s, v = zip(*c["val"])
        ax.plot(s, v, color=color, lw=lw, label=label)
    ax.plot([], [], color=LIGHT, lw=0.9, label="other post-norm variants")
    ax.axhline(UNIGRAM, color=INK, lw=0.5, alpha=0.5, zorder=0)
    ax.text(20000, UNIGRAM + 0.1, "unigram entropy", ha="right", va="bottom",
            fontsize=6.0, color=INK2)
    ax.set_xlim(0, 20000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xticklabels(["0", "5k", "10k", "15k", "20k"])
    ax.set_ylim(1.0, 6.6)
    ax.set_xlabel("step")
    ax.set_ylabel("validation CE (nats)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax.set_title("a  24 layers, K = 512")

    ck = [("al24_pnorm_s2000", MUTED, None), ("shiftpn24_s2000", ORANGE, "o"),
          ("shift24_s2000", ORANGE, None), ("coderes24_s2000", BLUE, None),
          ("al8_nopnorm_s2000", AQUA, None)]
    # (b) which blocks trained
    ax = axes[1]
    for name, color, mk in ck:
        per_layer(ax, load("ckpt", f"{name}.json"), "dW_rel_fc1", color, marker=mk)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3)
    ax.set_xticks([0, 8, 16, 23])
    ax.set_xlabel(r"block $\ell$")
    ax.set_ylabel(r"$\|W-W_0\|/\|W_0\|$, fc1")
    ax.set_title("b  weight change at 2k")

    # (c) token collapse
    ax = axes[2]
    for name, color, mk in ck:
        per_layer(ax, load("ckpt", f"{name}.json"), "tok_cos", color, marker=mk)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 8, 16, 23])
    ax.set_xlabel(r"block $\ell$")
    ax.set_ylabel("cosine between positions")
    ax.set_title("c  collapse at 2k")

    # (d) depth sweep at step 2000
    ax = axes[3]

    def at(run, step=2000):
        c = curves.get(run)
        return dict(c["val"]).get(step) if c else None

    sweeps = [
        ([(8, "li_8L_k512_pnorm"), (12, "li_12L_k512_pnorm"), (16, "li_16L_k512_pnorm"),
          (24, "al_500m_hard_k512_pnorm_bp_std")], "TopK, post-norm", MUTED, "o"),
        ([(8, "al_8L_hard_k512_nopnorm_bp_std"), (12, "li_12L_k512_nopnorm"),
          (16, "li_16L_k512_nopnorm"), (24, "al_500m_hard_k512_nopnorm_bp_std")],
         "TopK, no norm", AQUA, "s"),
        ([(8, "li_8L_k512_coderes_a1"), (12, "li_12L_k512_coderes_a1"),
          (16, "li_16L_k512_coderes_a1"), (24, "li_24L_k512_coderes_a1"),
          (48, "li_48L_k512_coderes_a1")],
         "code residual", BLUE, "o"),
    ]
    for pts, label, color, mk in sweeps:
        xy = []
        for L, run in pts:
            v = at(run)
            if v is None and curves.get(run) and curves[run]["val"]:
                v = curves[run]["val"][-1][1]  # died before 2k (overflow, ln V)
            if v is not None:
                xy.append((L, v))
        if xy:
            ax.plot(*zip(*xy), "-", marker=mk, color=color, ms=3.6, mec="white", mew=0.8,
                    label=label)
    ax.set_xticks([8, 16, 24, 48])
    ax.set_yscale("log")
    ax.set_ylim(1.4, 12)
    ax.set_yticks([1.5, 2, 3, 4, 6, 10])
    ax.set_yticklabels(["1.5", "2", "3", "4", "6", "10"])
    ax.minorticks_off()
    ax.axhline(UNIGRAM, color=INK, lw=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel("layers")
    ax.set_ylabel("validation CE at 2k")
    ax.legend(loc="center left", bbox_to_anchor=(0.40, 0.56))
    ax.set_title("d  depth")

    handles = [plt.Line2D([], [], color=c, lw=1.4, marker=m, ms=2.8, mfc="white", mew=0.9)
               for c, m in ((MUTED, None), (ORANGE, "o"), (ORANGE, None), (BLUE, None), (AQUA, None))]
    labels = ["TopK + post-norm (the failed run)", "value shift + post-norm",
              "value shift, no norm", "code residual", "TopK, no norm, 8 layers"]
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.62, -0.13),
               columnspacing=1.2, handlelength=1.6)
    save(fig, "fig_training")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_init()
    fig_training()
    print("wrote", OUT)
