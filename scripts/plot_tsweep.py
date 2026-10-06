"""Figures of the (reach, strength) grid for code-carried RBLapSum, 2026-10-05/06.

Reads docs/figures/tsweep/data/curves_tsweep.json (the ``ma_ts_*`` runs,
written by scripts/extract_val_curves.py) and the hard Top-K' ladder of the
code-carried K+J grid (docs/figures/kj-cr/data/curves_new.json); writes
docs/figures/tsweep/fig_ts_*.{pdf,png} and prints the step-10000 table.

Runs: ma_ts_{span|b}_t{025|050|100}_s{010|025|050|100}_k{32|256}_j{480|256}[_fo],
10k steps of the 20k schedule (train_guard --stop-step 10000), validation every 500.
The kernel width is T = tau * span (span = s_(K+1) - s_(K+J)) or T = tau * b
(b = s_(K+1)); the support strength s sets gamma = 2 s T / b per token, so the
(K+1)-th member's support gradient equals s times its read gradient.  The
``_fo`` runs use the first-order scope instead of the pool scope.

Colour conventions follow scripts/plot_cr_kj.py (docs/kj-vs-hard-topk-code-residual.tex):
the varied family is a blue ramp light to dark, hard baselines are dashed reds
light to dark with K', the reference run of the family is dotted grey.

  python scripts/plot_tsweep.py            # all figures and the table
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "docs", "figures", "tsweep", "data")
KJ = os.path.join(ROOT, "docs", "figures", "kj-cr", "data")
OUT = os.path.join(ROOT, "docs", "figures", "tsweep")

BLUES, REDS = cm.get_cmap("Blues"), cm.get_cmap("Reds")
B_RANGE, R_RANGE = (0.40, 0.95), (0.40, 0.92)
GREY = "0.40"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9,
                     "axes.grid": True, "grid.alpha": 0.35, "savefig.dpi": 200})

TAUS = [("025", 0.25), ("050", 0.5), ("100", 1.0)]
SS = [("010", 0.1), ("025", 0.25), ("050", 0.5), ("100", 1.0)]
CELLS = {32: 480, 256: 256}
HARD_KS = [32, 64, 128, 256, 512]
# the hard rungs drawn next to the zero line of each grid figure
EXTRA_HARD = {32: [64, 128], 256: [128, 512]}
MODE_LABEL = {"span": r"$T=\tau\,(s_{(K+1)}-s_{(K+J)})$", "b": r"$T=\tau\, s_{(K+1)}$"}
ARM_LABEL = {"pool": "pool scope", "fo": "first-order scope"}


def ramp(cmap, n, lo, hi):
    return [cmap(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]


def load(*parts):
    p = os.path.join(*parts)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def curves():
    return {**(load(KJ, "curves_new.json") or {}), **(load(DATA, "curves_tsweep.json") or {})}


def run_of(mode, tau, s, K, arm="pool"):
    tk = [k for k, v in TAUS if v == tau][0]
    sk = [k for k, v in SS if v == s][0]
    return f"ma_ts_{mode}_t{tk}_s{sk}_k{K}_j{CELLS[K]}" + ("_fo" if arm == "fo" else "")


def hard_of(K):
    return f"ma_cr_hard_k{K}_pnorm"


def ref_of(K, arm="pool"):
    """The T=2, gamma=1 run of the same scope: the configuration used so far."""
    return f"ma_cr_rbk_k{K}_j{CELLS[K]}_t2" + ("_fo" if arm == "fo" else "")


def series(cv, run):
    c = cv.get(run)
    return {} if c is None else dict((int(st), v) for st, v in c["val"])


def delta(cv, run, ref, max_step):
    d, h = series(cv, run), series(cv, ref)
    pts = [(st, v - h[st]) for st, v in sorted(d.items()) if st in h and st <= max_step]
    return list(zip(*pts)) if pts else ([], [])


# --------------------------------------------------------------------------- #
# grid figures: one panel per tau, the four s curves as a blue ramp
# --------------------------------------------------------------------------- #
def fig_grid(mode, K, arm="pool", max_step=10000):
    cv = curves()
    J = CELLS[K]
    scol = dict(zip([s for _, s in SS], ramp(BLUES, len(SS), *B_RANGE)))
    hcol = dict(zip(EXTRA_HARD[K], ramp(REDS, len(EXTRA_HARD[K]), *R_RANGE)))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
    for ax, (tk, tau) in zip(axes, TAUS):
        for kp in EXTRA_HARD[K]:
            x, y = delta(cv, hard_of(kp), hard_of(K), max_step)
            ax.plot(x, y, "--", color=hcol[kp], lw=1.6)
        x, y = delta(cv, ref_of(K, arm), hard_of(K), max_step)
        ax.plot(x, y, ":", color=GREY, lw=1.8)
        for _, s in SS:
            x, y = delta(cv, run_of(mode, tau, s, K, arm), hard_of(K), max_step)
            ax.plot(x, y, "-", color=scol[s], lw=2.0, marker="o", ms=3, mfc="white", mew=0.8)
        ax.axhline(0, color="0.15", lw=1.4)
        ax.set_title(rf"$\tau = {tau:g}$")
        ax.set_xlabel("step")
        ax.set_xlim(0, max_step)
    axes[0].set_ylabel(rf"val CE $-$ hard Top-{K}")
    hd = [Line2D([0], [0], color="0.15", lw=1.4, label=f"hard Top-{K}")]
    hd += [Line2D([0], [0], color=hcol[kp], ls="--", lw=1.6, label=f"hard Top-{kp}") for kp in EXTRA_HARD[K]]
    hd += [Line2D([0], [0], color=GREY, ls=":", lw=1.8,
                  label=rf"{ARM_LABEL[arm]}, $T=2$, $\gamma=1$")]
    hd += [Line2D([0], [0], color=scol[s], lw=2.0, label=rf"$s={s:g}$") for _, s in SS]
    axes[2].legend(handles=hd, loc="best", framealpha=0.93, ncol=2)
    fig.suptitle(rf"$K={K}$, $J={J}$, {MODE_LABEL[mode]}, $\gamma=2sT/b$, {ARM_LABEL[arm]}",
                 y=1.01, fontsize=12)
    save(fig, f"fig_ts_{arm}_{mode}_k{K}")


# --------------------------------------------------------------------------- #
# best cells against the hard ladder
# --------------------------------------------------------------------------- #
def best_cells(arm="pool", step=10000, modes=("span", "b")):
    cv = curves()
    best = {}
    for mode in modes:
        for K in CELLS:
            cand = [(series(cv, run_of(mode, tau, s, K, arm)).get(step), tau, s)
                    for _, tau in TAUS for _, s in SS]
            cand = [c for c in cand if c[0] is not None]
            if cand:
                best[(mode, K)] = min(cand)
    return best


def fig_best(arm="pool", modes=("span", "b"), max_step=10000, step=10000, absolute=False):
    cv = curves()
    best = best_cells(arm, step, modes)
    hcol = dict(zip(HARD_KS, ramp(REDS, len(HARD_KS), *R_RANGE)))
    ccol = {32: BLUES(0.60), 256: BLUES(0.95)}
    fig, axes = plt.subplots(1, len(modes), figsize=(7.5 * len(modes), 4.8), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, mode in zip(axes, modes):
        for kp in HARD_KS:
            if absolute:
                d = series(cv, hard_of(kp))
                x, y = zip(*[(st, v) for st, v in sorted(d.items()) if 500 <= st <= max_step])
            else:
                x, y = delta(cv, hard_of(kp), hard_of(512), max_step)
            ax.plot(x, y, "--", color=hcol[kp], lw=1.7)
        for K in CELLS:
            if (mode, K) not in best:
                continue
            _, tau, s = best[(mode, K)]
            r = run_of(mode, tau, s, K, arm)
            if absolute:
                d = series(cv, r)
                x, y = zip(*[(st, v) for st, v in sorted(d.items()) if 500 <= st <= max_step])
            else:
                x, y = delta(cv, r, hard_of(512), max_step)
            ax.plot(x, y, "-", color=ccol[K], lw=2.2, marker="o", ms=3, mfc="white", mew=0.8,
                    label=rf"$K={K}$, $J={CELLS[K]}$: $\tau={tau:g}$, $s={s:g}$")
        if not absolute:
            ax.axhline(0, color="0.15", lw=1.0)
        ax.set_title(MODE_LABEL[mode] + rf", best cells at step {step}, {ARM_LABEL[arm]}")
        ax.set_xlabel("step")
        ax.set_xlim(500 if absolute else 0, max_step)
        hd = [Line2D([0], [0], color=hcol[kp], ls="--", lw=1.7, label=f"hard Top-{kp}") for kp in HARD_KS]
        hd += ax.get_legend_handles_labels()[0]
        lo, hi = ax.get_ylim()                      # headroom so the legend covers no curve
        ax.set_ylim(lo, hi + 0.30 * (hi - lo))
        ax.legend(handles=hd, loc="upper right", framealpha=0.93, ncol=2)
    axes[0].set_ylabel("val CE" if absolute else r"val CE $-$ hard Top-512")
    save(fig, f"fig_ts_{arm}_best" + ("_abs" if absolute else ""))


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table(arm="pool", step=10000, modes=("span", "b")):
    cv = curves()
    rows = []
    for mode in modes:
        for K in CELLS:
            h = series(cv, hard_of(K)).get(step)
            for _, tau in TAUS:
                for _, s in SS:
                    r = run_of(mode, tau, s, K, arm)
                    c = cv.get(r)
                    v = series(cv, r).get(step)
                    dg = c["diag"] if c else []
                    last = dg[-1] if dg else [None] * 5
                    rows.append((mode, K, tau, s, v, None if (v is None or h is None) else v - h,
                                 last[1], last[2], last[3], c.get("diverged") if c else None))
    return rows


def print_table(arm, modes):
    print(f"\n== {ARM_LABEL[arm]}: val CE at 10000, minus hard Top-K; T_eff, gamma, b at 10k")
    print(f"{'mode':5} {'K':>4} {'tau':>5} {'s':>5} {'val CE':>8} {'- hard':>8} {'T_eff':>8} {'gamma':>7} {'b':>8}  div")
    for mode, K, tau, s, v, dv, T, g, b, div in table(arm, modes=modes):
        f = lambda x, w=8, p=4: f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f"{mode:5} {K:>4} {tau:>5g} {s:>5g} {f(v)} {f(dv)} {f(T, 8, 3)} {f(g, 7, 3)} {f(b, 8, 3)}  {'yes' if div else ''}")
    cv = curves()
    for (mode, K), (v, tau, s) in sorted(best_cells(arm, modes=modes).items()):
        h = series(cv, hard_of(K)).get(10000)
        print(f"best {mode:5} K={K:>3}: tau={tau:g} s={s:g}  val {v:.4f}  ({v - h:+.4f} vs hard Top-{K})")


def main():
    os.makedirs(OUT, exist_ok=True)
    cv = curves()
    for mode in ("span", "b"):
        for K in CELLS:
            fig_grid(mode, K, "pool")
    fig_best("pool")
    fig_best("pool", absolute=True)
    print_table("pool", ("span", "b"))
    if any(k.endswith("_fo") and k.startswith("ma_ts_") for k in cv):
        for K in CELLS:
            fig_grid("span", K, "fo")
        fig_best("fo", modes=("span",))
        fig_best("fo", modes=("span",), absolute=True)
        print_table("fo", ("span",))


if __name__ == "__main__":
    main()
