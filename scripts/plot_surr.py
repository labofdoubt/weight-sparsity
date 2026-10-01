"""Figures for docs/rblapsum-code-residual*.tex (RBLapSum in a code-residual stack).

Reads docs/figures/surr/data (analysis/surrogate_probe.py JSONs at
initialization, code_usage JSONs at checkpoints, the onset series and the curve
extract) and writes docs/figures/surr/fig_*.pdf/png.

  python scripts/plot_surr.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "surr", "data")
OUT = os.path.join(ROOT, "docs", "figures", "surr")

# the palette and rc of scripts/plot_depth.py
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.2,
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

HARD_RUN = "li_sr2_k32_j224_pool_t1_ss0"     # support scale 0: the exact hard backward


def load(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def val_at(curves, run, step):
    c = curves.get(run)
    if c is None:
        return None
    d = dict((int(s), v) for s, v in c["val"])
    return d.get(step)


# --------------------------------------------------------------------------- #
# Figure 1: the two mechanisms
# --------------------------------------------------------------------------- #
INIT_PROFILES = [  # (probe json, label, color, lw)
    ("hard.json", "hard", INK, 1.1),
    ("pool_t1.json", "unmodified", ORANGE, 1.4),
    ("upd_t1.json", "into the update", AQUA, 1.2),
    ("fo_t1.json", "first order", BLUE, 1.4),
    ("pool_t1_ss03.json", r"unmodified, $\gamma{=}0.3$", YELLOW, 1.2),
]

ONSET_GATES = [(6, BLUE), (12, AQUA), (18, YELLOW), (24, ORANGE)]


def onset_series(run):
    pts = []
    for f in glob.glob(os.path.join(DATA, "onset", f"{run}_ckpt_step*.json")):
        d = json.load(open(f))
        pts.append((int(d["step"]), {g["gate"]: g for g in d["gates"]}))
    pts.sort(key=lambda p: p[0])
    return pts


def fig_mechanisms():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"wspace": 0.38})

    # (a) gradient reaching the carried code, at initialization
    ax = axes[0]
    hard = [g["code_grad_rms"] for g in load("init2", "hard.json")["gates"]]
    for name, lab, color, lw in INIT_PROFILES:
        d = load("init2", name)
        if d is None:
            continue
        y = [g["code_grad_rms"] / h for g, h in zip(d["gates"], hard)]
        ax.plot(range(len(y)), y, "-", color=color, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel(r"gate $\ell$ (0 = entry)")
    ax.set_ylabel(r"RMS $\partial\mathcal{L}/\partial c_\ell$, relative to hard")
    ax.legend(loc="upper right", fontsize=5.6)
    ax.set_title("a  backward along the code, step 0")

    # (b) the onset: always-on features in the unmodified-routing collapse
    ax = axes[1]
    pts = onset_series("on_upd_t1")
    for gate, color in ONSET_GATES:
        if not pts:
            break
        ys = [g[gate]["energy_half"] for _, g in pts]
        ax.plot([s for s, _ in pts], ys, "-o", ms=2.4, color=color, mfc="white", mew=0.8)
        if gate in (12, 24):
            ax.text(pts[-1][0] + 18, ys[-1], f"gate {gate}", fontsize=5.8, color=INK2, va="center")
    if pts:
        ax.text(pts[-1][0] + 18, 0.985, "gates 6, 18", fontsize=5.8, color=INK2, va="center")
    ref = onset_series("on_ss0")
    if ref:
        ys = [max(g[k]["energy_half"] for k, _ in ONSET_GATES) for _, g in ref]
        ax.plot([s for s, _ in ref], ys, "-", color=INK, lw=0.9)
        ax.text(ref[-1][0] + 18, ys[-1], "hard", fontsize=5.8, color=INK2, va="center")
    ax.set_xlim(20, 700)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel("step")
    ax.set_ylabel("code energy in always-on features")
    ax.set_title("b  always-on features, routed run")

    # (c) dose: initial surrogate energy vs the outcome at step 2000
    ax = axes[2]
    curves = load("curves.json") or {}
    ref = val_at(curves, HARD_RUN, 2000)
    rows = [  # run, probe at init, marker, color
        ("li_sr2_k32_j224_pool_t1_ss003", "pool_t1_ss003.json", "o", ORANGE),
        ("li_sr_k32_j224_t1_ss01", "pool_t1_ss01.json", "o", ORANGE),
        ("li_sr2_k32_j224_pool_t1_ss03", "pool_t1_ss03.json", "o", ORANGE),
        ("li_sr_k32_j224_t1_plain", "pool_t1.json", "o", ORANGE),
        ("li_sr2_k32_j224_upd_t1_ss03", "upd_t1_ss03.json", "s", AQUA),
        ("li_sr2_k32_j224_upd_rel1", "upd_rel1.json", "s", AQUA),
        ("li_sr_k32_j224_t1_local", "upd_t1.json", "s", AQUA),
        ("li_sr_k32_j224_t1_inactive", "inactive_t1.json", "D", MAGENTA),
        ("li_sr4_k32_j224_fo_t1_ss03", "fo_t1_ss03.json", "^", BLUE),
        ("li_sr4_k32_j224_fo_t1", "fo_t1.json", "^", BLUE),
    ]
    for run, probe, marker, color in rows:
        d = load("init2", probe)
        v = val_at(curves, run, 2000)
        if d is None or v is None or ref is None:
            continue
        sig = sum(g["sigma"] for g in d["gates"]) / len(d["gates"])
        dv = v - ref
        clipped = dv > 0.35
        ax.plot([sig], [min(dv, 0.35)], marker, ms=4.2, color=color, mec="white", mew=0.6,
                alpha=0.55 if clipped else 1.0)
    ax.axhline(0, color=INK, lw=0.7)
    for marker, color, lab in (("o", ORANGE, "unmodified routing"),
                               ("s", AQUA, "into the update"),
                               ("D", MAGENTA, "inactive members only"),
                               ("^", BLUE, "first order")):
        ax.plot([], [], marker, color=color, mec="white", ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 0.37)
    ax.set_xlabel(r"surrogate/hard energy $\sigma$ at step 0")
    ax.set_ylabel("val CE at 2k $-$ hard TopK")
    ax.legend(loc="upper left", fontsize=5.8)
    ax.set_title("c  dose and outcome")
    save(fig, "fig_mechanisms")


# --------------------------------------------------------------------------- #
# Figure 2: training
# --------------------------------------------------------------------------- #
SCREENS = [  # run, label, color, marker, style
    ("li_sr4_k32_j224_fo_t1", r"first order, $\gamma{=}1$", BLUE, "o", "-"),
    ("li_sr4_k32_j224_fo_t1_ss03", r"first order, $\gamma{=}0.3$", BLUE, "o", "--"),
    ("li_sr2_k32_j224_pool_t1_ss03", r"unmodified, $\gamma{=}0.3$", YELLOW, "s", "-"),
    ("li_sr_k32_j224_t1_ss01", r"unmodified, $\gamma{=}0.1$", YELLOW, "s", "--"),
    ("li_sr_k32_j224_t1_inactive", r"inactive members only", MAGENTA, "D", "-"),
    ("li_sr2_k32_j224_upd_t1_ss03", r"into the update, $\gamma{=}0.3$", AQUA, "^", "-"),
]
HARD_256 = "li_sr5_k256_j256_ss0"
SCREENS_256 = [  # run, label, color, marker, style
    ("li_sr5_k256_j256_fo_t1", r"first order, $\gamma{=}1$", BLUE, "o", "-"),
    ("li_sr7_k256_j256_fo_t1_ss03", r"first order, $\gamma{=}0.3$", BLUE, "o", "--"),
    ("li_sr8_k256_j256_foinact_t1", r"first order, inactive only", MAGENTA, "D", "-"),
    ("li_sr8_k256_j256_fo_rel03", r"first order, $T{=}0.3b$", AQUA, "^", "-"),
    ("li_sr7_k256_j256_pool_t1_ss03", r"unmodified, $\gamma{=}0.3$", YELLOW, "s", "-"),
]
LONG = [  # run, label, color, style
    (HARD_RUN, "hard backward", INK, "-"),
    ("li_sr4_k32_j224_fo_t1", r"first order, $\gamma{=}1$", BLUE, "-"),
    ("li_sr2_k32_j224_pool_t1_ss03", r"unmodified, $\gamma{=}0.3$", YELLOW, "-"),
    ("li_sr_k32_j224_t1_ss01", r"unmodified, $\gamma{=}0.1$", YELLOW, "--"),
]
DYN = [  # run, label, color, style
    (HARD_RUN, "hard backward", INK, "-"),
    ("li_sr_k32_j224_t1_plain", r"unmodified, $\gamma{=}1$", ORANGE, "-"),
    ("li_sr_k32_j224_t1_local", r"into the update, $\gamma{=}1$", AQUA, "-"),
    ("li_sr2_k32_j224_pool_t1_ss03", r"unmodified, $\gamma{=}0.3$", YELLOW, "-"),
    ("li_sr4_k32_j224_fo_t1", r"first order, $\gamma{=}1$", BLUE, "-"),
]


def delta_panel(ax, curves, runs, hard_run, max_step=2000):
    hard = dict((int(s), v) for s, v in curves.get(hard_run, {"val": []})["val"])
    for run, lab, color, marker, ls in runs:
        c = curves.get(run)
        if c is None:
            continue
        pts = [(int(s), v - hard[int(s)]) for s, v in c["val"]
               if int(s) <= max_step and int(s) in hard]
        if pts:
            ax.plot(*zip(*pts), ls, color=color, lw=1.2, marker=marker, ms=2.6,
                    mfc="white", mew=0.8, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([500, 1000, 1500, 2000])
    ax.set_xlabel("step")
    ax.set_ylabel("val CE $-$ hard backward")


def fig_training():
    curves = load("curves.json") or {}
    hard = dict((int(s), v) for s, v in curves.get(HARD_RUN, {"val": []})["val"])
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.95),
                             gridspec_kw={"wspace": 0.5, "width_ratios": [1.1, 1.1, 1.1, 1]})

    ax = axes[0]
    delta_panel(ax, curves, SCREENS, HARD_RUN)
    ax.set_ylim(-0.09, 0.16)
    ax.legend(loc="upper left", fontsize=4.9)
    ax.set_title(r"a  $K{=}32$, $J{=}224$")

    ax = axes[1]
    delta_panel(ax, curves, SCREENS_256, HARD_256)
    ax.set_ylim(-0.09, 0.16)
    ax.legend(loc="lower left", fontsize=4.9)
    ax.set_title(r"b  $K{=}256$, $J{=}256$")

    # (c) the long runs, K=32
    ax = axes[2]
    for run, lab, color, ls in LONG:
        c = curves.get(run)
        if c is None or run == HARD_RUN:
            continue
        pts = [(int(s), v - hard[int(s)]) for s, v in c["val"] if int(s) in hard]
        if pts:
            ax.plot(*zip(*pts), ls, color=color, lw=1.2, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.1, 0.03)
    ax.set_xlabel("step")
    ax.set_ylabel("val CE $-$ hard backward")
    ax.legend(loc="upper right", fontsize=4.9)
    ax.set_title(r"c  long runs, $K{=}32$")

    # (d) scale of the code
    ax = axes[3]
    for run, lab, color, ls in DYN:
        c = curves.get(run)
        if c is None:
            continue
        cv = c["curves"]
        ys = [(s, y) for s, y in zip(cv["step"], cv["b"]) if y is not None and s <= 2000]
        if ys:
            ax.plot(*zip(*ys), ls, color=color, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"rank boundary $b$ (mean over gates)")
    ax.text(1960, 1700, r"into the update, $\gamma{=}1$", fontsize=5.4, color=INK2, ha="right")
    ax.text(1960, 140, r"unmodified, $\gamma{=}1$", fontsize=5.4, color=INK2, ha="right")
    ax.text(1960, 4.3, "hard, first order,\n" + r"unmodified $\gamma{=}0.3$", fontsize=5.4,
            color=INK2, ha="right", va="bottom")
    ax.set_ylim(1.6, 3000)
    ax.set_title(r"d  scale of the code, $K{=}32$")
    save(fig, "fig_training")


if __name__ == "__main__":
    fig_mechanisms()
    fig_training()
    print("wrote", sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "fig_*.pdf"))))
