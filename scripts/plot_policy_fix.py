"""Figures of the diagnosis-and-fix part of the policy_cr note.

Inputs (docs/figures/policy-cr/data/):
  curves_vienna.json   scripts/extract_policy_curves.py over the vienna runs: the eight
                       likelihood-ratio campaign runs vi_pol_rs_*, the gamma = 0 controls
                       vi_salv_*_g0, the Rao-Blackwell starts vi_rb_*
  curves_zurich.json   the same over the zurich runs (zu_rbfo_*, zu_salv_*), optional
  probe/probe_*.json   scripts/policy_grad_probe.py (three states of the (32, 480) cell)
  ../kj-cr/data/curves_new.json   madrid references (RBLapSum T=2, hard Top-K' post-norm)

Figures (docs/figures/policy-cr/):
  pf_failure.png   clean validation CE of the eight likelihood-ratio runs against RBLapSum
                   T=2 and hard Top-K, one panel per cell; the gamma = 0 controls
  pf_gradnorm.png  gradient norm before clipping against step: likelihood ratio, gamma = 0,
                   Rao-Blackwell full / first-order scope, RBLapSum and hard references
  pf_probe.png     per-layer noise (per-draw sqrt tr Cov) and signal (bias-corrected
                   |E g|) of the selection-gradient estimators against |g_value|, at the
                   three probed states
  pf_fix.png       the Rao-Blackwell first-order run against RBLapSum T=2, hard Top-32,
                   hard Top-512, the likelihood-ratio run and gamma = 0 (left: all steps,
                   right: tight scale)
  pf_rb_blocks.png per-block exchange fraction and Rao-Blackwell score-gradient RMS of
                   the first-order run against step
  pf_salvage.png   the gamma grid {0, 0.1, 1, 10} x gamma*: training and validation CE

    python scripts/plot_policy_fix.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
DATA = os.path.join(HERE, "policy-cr", "data")
OUT = os.path.join(HERE, "policy-cr")
CELLS = [(32, 32), (32, 480), (128, 128), (256, 256)]
C_CONST, C_ANNEAL = "#1f4e9c", "#4fa3d1"
C_RB, C_HARD, C_HARD2 = "0.35", "#c0392b", "#e59866"
C_G0, C_FIX, C_FULL = "#2ca02c", "#7b3294", "#8c510a"
VIRIDIS = cm.get_cmap("viridis")


def load(name, default=None):
    try:
        with open(os.path.join(DATA, name)) as f:
            return json.load(f)
    except OSError:
        return default


def save(fig, fn):
    fig.tight_layout()
    p = os.path.join(OUT, fn)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(p)


def val(rec, k=1):
    return [p[0] for p in rec["val"]], [p[k] for p in rec["val"]]


def fig_failure(cv, ref):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    for ax, (k, j) in zip(axes, CELLS):
        for name, rec in cv.items():
            c = rec["cfg"]
            if (c["k"], c["j"]) != (k, j):
                continue
            if name.startswith("vi_pol_rs"):
                arm = c["schedule"] == "exponential"
                ax.plot(*val(rec), "-o", ms=3, color=C_ANNEAL if arm else C_CONST, lw=1.8,
                        label=f"likelihood ratio, {'annealed' if arm else 'constant'}")
            elif name.startswith("vi_salv") and rec["val"]:
                ax.plot(*val(rec), "s", ms=7, color=C_G0, label="$\\gamma = 0$ (value path only)")
        for run, color, ls, lab in ((f"ma_cr_rbk_k{k}_j{j}_t2", C_RB, ":", "RBLapSum $T=2$"),
                                    (f"ma_cr_hard_k{k}_pnorm", C_HARD, "--", f"hard Top-${k}$")):
            if run in ref:
                s, v = zip(*[(p[0], p[1]) for p in ref[run]["val"] if p[0] <= 3000])
                ax.plot(s, v, ls, color=color, lw=1.8, label=lab)
        ax.axhline(5.9, color="0.6", lw=0.8, ls="-.")
        ax.text(150, 5.95, "unigram level", fontsize=7, color="0.4")
        ax.set_xlim(0, 3000)
        ax.set_ylim(1.7, 7.8)
        ax.set_title(f"$(K, J) = ({k}, {j})$")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("clean Top-$K$ validation CE (nats)")
    save(fig, "pf_failure.png")


def fig_gradnorm(cv, cz):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, rec in {**cv, **(cz or {})}.items():
        tr = rec["train"]
        if not tr:
            continue
        s = [p[0] for p in tr]
        g = [p[3] for p in tr]
        if name.startswith("vi_pol_rs"):
            ax.plot(s, g, "-", color=C_CONST, lw=0.5, alpha=0.5)
        elif "salv" in name and name.endswith("_g0"):
            ax.plot(s, g, "-", color=C_G0, lw=1.2)
        elif name.startswith("vi_rb") and "full" in name:
            ax.plot(s, g, "-", color=C_FULL, lw=1.5)
        elif "rbfo" in name or name.endswith("_fo"):
            key = "project" if "_proj_" in name else ("frozen" if "_frozen_" in name else "through")
            ax.plot(s, g, "-", color=WIDTH_STYLE[key][0] if name.startswith("zu_") else C_FIX,
                    lw=1.0)
    refdir = os.path.join(DATA, "ref")
    for run, color in (("ma_cr_rbk_k32_j480_t2", C_RB), ("ma_cr_hard_k32_pnorm", C_HARD)):
        path = os.path.join(refdir, run, "metrics.jsonl")
        if os.path.isfile(path):
            rows = [json.loads(line) for line in open(path)]
            tr = [(r["step"], r["train/grad_norm"]) for r in rows if "train/grad_norm" in r]
            ax.plot([a for a, _ in tr], [b for _, b in tr], ":", color=color, lw=1.0)
    ax.axhline(1.0, color="k", lw=1, ls="-.")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(1, 20000)
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm before clipping")
    from matplotlib.lines import Line2D
    hd = [Line2D([0], [0], color=C_CONST, lw=1.2, label="likelihood ratio (8 runs)"),
          Line2D([0], [0], color=C_FULL, lw=1.5, label="Rao-Blackwell, full scope"),
          Line2D([0], [0], color="#7b3294", lw=1.2, label="Rao-Blackwell, first order, projected"),
          Line2D([0], [0], color="#c2a5cf", lw=1.2, label="Rao-Blackwell, first order, through"),
          Line2D([0], [0], color="#e08214", lw=1.2, label="Rao-Blackwell, first order, frozen"),
          Line2D([0], [0], color=C_G0, lw=1.2, label="$\\gamma = 0$"),
          Line2D([0], [0], color=C_RB, ls=":", lw=1.2, label="RBLapSum $T=2$, (32, 480)"),
          Line2D([0], [0], color=C_HARD, ls=":", lw=1.2, label="hard Top-32"),
          Line2D([0], [0], color="k", ls="-.", lw=1, label="clip threshold")]
    ax.legend(handles=hd, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "pf_gradnorm.png")


def fig_probe(probes):
    states = [(k, lab) for k, lab in (("init", "initialization"),
                                      ("policyckpt", "likelihood-ratio run, step 2000"),
                                      ("rblapsumckpt", "RBLapSum $T=2$ run, step 2000"))
              if k in probes]
    fig, axes = plt.subplots(1, len(states), figsize=(6 * len(states), 4.8), sharey=True)
    axes = [axes] if len(states) == 1 else axes
    for ax, (key, lab) in zip(axes, states):
        rep = probes[key]
        groups = [g for g in rep["groups"] if g.startswith("block") or g == "embed/other"]
        order = sorted(groups, key=lambda g: (g == "embed/other", g))
        xs = range(len(order))
        gv = [rep["groups"][g]["val"]["g_value_norm"] for g in order]
        ax.plot(xs, gv, "k-", lw=2, label="$|g_{\\mathrm{value}}|$ (clean)")
        for est, color, lab2 in (("lr", C_CONST, "likelihood ratio"),
                                 ("rb", C_FULL, "Rao-Blackwell, full"),
                                 ("rbfo", C_FIX, "Rao-Blackwell, first order")):
            if est not in rep["groups"][order[0]]:
                continue
            nan = float("nan")
            noise = [rep["groups"][g][est]["noise"] or nan for g in order]
            ax.plot(xs, noise, "-", color=color, lw=1.6, label=f"{lab2}: noise")
            if est != "lr":   # the LR mean is not resolved by 64 draws: no signal line
                sig = [rep["groups"][g][est]["signal"] or nan for g in order]
                ax.plot(xs, sig, "--", color=color, lw=1.4, label=f"{lab2}: signal")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([g.replace("block", "b").replace(".branches", " br")
                            .replace(".encoder", " enc").replace(".decoder", " dec")
                            .replace(".bottleneck_other", " gate")
                            .replace("embed/other", "emb") for g in order],
                           rotation=90, fontsize=7)
        ax.set_title(f"{lab}\nclean CE {rep['clean_ce']:.3f}, $\\gamma_* = {rep.get('gamma_star', float('nan')):.2g}$",
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("gradient norm per parameter group")
    axes[0].legend(fontsize=7)
    save(fig, "pf_probe.png")


WIDTH_STYLE = {"project": ("#7b3294", "-", 2.4, "Rao-Blackwell, first order, projected width"),
               "through": ("#c2a5cf", "-", 1.6, "Rao-Blackwell, first order, span differentiated"),
               "frozen": ("#e08214", "-", 1.6, "Rao-Blackwell, first order, frozen width")}


def rb_runs(cz):
    """zurich Rao-Blackwell first-order runs keyed by width gradient."""
    out = {}
    for name, rec in (cz or {}).items():
        if not name.startswith("zu_rbfo"):
            continue
        key = "project" if "_proj_" in name else ("frozen" if "_frozen_" in name else "through")
        out[key] = (name, rec)
    return out


def fig_fix(cv, cz, ref):
    rbs = rb_runs(cz)
    if not rbs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, zoom in zip(axes, (False, True)):
        for run, color, ls, lab in (("ma_cr_rbk_k32_j480_t2", C_RB, ":", "RBLapSum $T=2$, (32, 480)"),
                                    ("ma_cr_hard_k32_pnorm", C_HARD, "--", "hard Top-32"),
                                    ("ma_cr_hard_k512_pnorm", C_HARD2, "--", "hard Top-512")):
            if run in ref:
                ax.plot(*zip(*[(p[0], p[1]) for p in ref[run]["val"]]), ls, color=color, lw=1.8,
                        label=lab)
        if not zoom:
            lr = cv.get("vi_pol_rs_k32_j480_t0p16_const")
            if lr:
                ax.plot(*val(lr), "-o", ms=3, color=C_CONST, lw=1.5,
                        label="likelihood ratio (stopped at 2540)")
        for name, rec in {**cv, **(cz or {})}.items():
            if "k32_j480" in name and name.endswith("_g0") and rec["val"]:
                ax.plot(*val(rec), "s", ms=7, color=C_G0,
                        label="$\\gamma = 0$" if "vi_" in name else None)
        for key in ("frozen", "through", "project"):
            if key in rbs:
                color, ls, lw, lab = WIDTH_STYLE[key]
                ax.plot(*val(rbs[key][1]), ls, color=color, lw=lw, label=lab)
        ax.set_xlim(0, 20000)
        if zoom:
            tail = [p[1] for p in rbs.get("project", ("", {"val": []}))[1]["val"] if p[0] >= 3000]
            refs = [p[1] for run in ("ma_cr_rbk_k32_j480_t2", "ma_cr_hard_k32_pnorm",
                                     "ma_cr_hard_k512_pnorm") if run in ref
                    for p in ref[run]["val"] if p[0] >= 3000]
            if tail:
                ax.set_ylim(min(tail + refs) - 0.01, max(tail + refs) + 0.01)
                ax.set_xlim(3000, 20000)
            ax.set_title("from step 3000")
        else:
            ax.set_ylim(1.3, 3.6)
            ax.set_title("all steps")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("clean Top-$K$ validation CE (nats)")
    axes[0].legend(fontsize=7.5)
    save(fig, "pf_fix.png")


def fig_span(cv, cz):
    """Median pool span over blocks against step: the frozen-width runaway."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    lr = cv.get("vi_pol_rs_k32_j480_t0p16_const")
    series = []
    if lr:
        series.append((lr, C_CONST, "-", 1.2, "likelihood ratio, frozen width"))
    g0 = cv.get("vi_salv_k32_j480_g0")
    if g0:
        series.append((g0, C_G0, "-", 1.2, "$\\gamma = 0$"))
    for key, (name, rec) in rb_runs(cz).items():
        color, ls, lw, lab = WIDTH_STYLE[key]
        series.append((rec, color, ls, lw, lab))
    for rec, color, ls, lw, lab in series:
        sp = rec["blocks"].get("span", [])
        if sp:
            ax.plot([p[0] for p in sp], [sorted(p[1])[len(p[1]) // 2] for p in sp], ls,
                    color=color, lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, 3000)
    ax.set_xlabel("training step")
    ax.set_ylabel("pool span $s_{(K+1)} - s_{(K+J)}$, median over blocks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "pf_span.png")


def fig_rb_blocks(cz):
    rbs = rb_runs(cz)
    if "project" not in rbs:
        return
    name, rec = rbs["project"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, key, lab, log in ((axes[0], "exchange", "exchange $q_\\ell$", False),
                              (axes[1], "span", "pool span $s_{(K+1)} - s_{(K+J)}$", True),
                              (axes[2], "rb_grad", "Rao-Blackwell score-gradient RMS", True)):
        series = rec["blocks"].get(key, [])
        if not series:
            continue
        steps = [p[0] for p in series]
        nb = len(series[0][1])
        for b in range(nb):
            ax.plot(steps, [p[1][b] for p in series], color=VIRIDIS(b / max(1, nb - 1)), lw=1.0,
                    label=f"block {b}")
        if log:
            ax.set_yscale("log")
        if key == "exchange":
            ax.axhspan(0.15, 0.20, color="0.88", zorder=0)
        ax.set_ylabel(lab)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, "pf_rb_blocks.png")


def fig_salvage(cz, cv, ref):
    runs = sorted((n, r) for n, r in (cz or {}).items() if "_salv_" in n)
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    cmap = cm.get_cmap("plasma")
    for i, (name, rec) in enumerate(runs):
        tag = name.rsplit("_", 1)[-1]
        color = cmap(0.1 + 0.8 * i / max(1, len(runs) - 1))
        tr = rec["train"]
        axes[0].plot([p[0] for p in tr], [p[2] if p[2] is not None else p[1] for p in tr], "-",
                     color=color, lw=1.5, label=f"{tag}")
        axes[1].plot([p[0] for p in tr], [p[3] for p in tr], "-", color=color, lw=1.0, label=tag)
    path = os.path.join(DATA, "ref", "ma_cr_rbk_k32_j480_t2", "metrics.jsonl")
    if os.path.isfile(path):
        rows = [json.loads(line) for line in open(path)]
        tr = [(r["step"], r["train/ce"], r["train/grad_norm"]) for r in rows
              if "train/ce" in r and r["step"] <= 1000]
        axes[0].plot([a for a, _, _ in tr], [b for _, b, _ in tr], ":", color=C_RB, lw=1.8,
                     label="RBLapSum $T=2$")
        axes[1].plot([a for a, _, _ in tr], [c for _, _, c in tr], ":", color=C_RB, lw=1.8)
    lr = cv.get("vi_pol_rs_k32_j480_t0p16_const")
    if lr:
        tr = [p for p in lr["train"] if p[0] <= 1000]
        axes[0].plot([p[0] for p in tr], [p[2] if p[2] is not None else p[1] for p in tr], "-",
                     color=C_CONST, lw=1.0, alpha=0.6, label="$\\gamma = 1$ (campaign run)")
    axes[0].set_ylabel("training CE, noise-off probe (nats)")
    axes[1].set_ylabel("gradient norm before clipping")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    save(fig, "pf_salvage.png")


def main():
    cv = load("curves_vienna.json", {})
    cz = load("curves_zurich.json", {})
    ref = load(os.path.join("..", "..", "kj-cr", "data", "curves_new.json"), {})
    probes = {}
    for key in ("init", "policyckpt", "rblapsumckpt"):
        p = load(os.path.join("probe", f"probe_{key}_k32_j480.json"))
        if p:
            probes[key] = p
    fig_failure(cv, ref)
    fig_gradnorm(cv, cz)
    if probes:
        fig_probe(probes)
    fig_fix(cv, cz, ref)
    fig_span(cv, cz)
    fig_rb_blocks(cz)
    fig_salvage(cz, cv, ref)


if __name__ == "__main__":
    main()
