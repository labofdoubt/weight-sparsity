"""Figures and tables of docs/gumbel-vs-hard-topk-code-residual.tex.

  python scripts/plot_gumbel_cr_report.py compact OUT.json CURVES.json [CURVES.json ...]
  python scripts/plot_gumbel_cr_report.py plot RUNS.json FIGDIR

``compact`` merges the per-box files of ``scripts/plot_gumbel_cr.py extract``
into one: abs-TopK runs only, validation curves for every run, the logged
diagnostics for the 20000-step runs only.

Variants (which bottlenecks draw a Gumbel support in a training forward):
"" all 8, "3f" 3 of 8 drawn once per forward, "3t" 3 of 8 drawn per token.

Figures (vertical axes capped at 2.0 nats, as in the K+J notes):
  hard_ladder.png     hard Top-K, K = 16 ... 256: whole run, and steps 8000-20000
  active_k<K>.png     one active count K, one column per variant: every 20000-step
                      Gumbel cell with that K (solid, blues light-to-dark with J)
                      against all hard Top-K' (dashed, reds light-to-dark with K');
                      top row the whole run, bottom row steps 10000-20000
  pools.png           one panel per pool K+J: hard Top-(K+J) (black dashed), the
                      other hard runs (light red dashed) and every Gumbel cell with
                      that pool (color: K; line style: variant), steps 10000-20000
  search.png          the (tau, lambda) search at step 2500: val/ce minus hard Top-K
                      at the same step, mean over the nine cells (lines: tau;
                      band: min-max over cells), one panel per variant
  exchange.png        share of the drawn support outside TopK during the 20000-step
                      runs (all-bottleneck variant)
"""
import json
import math
import sys

STOP = 2500
VARIANTS = [("", "all 8 bottlenecks sample"), ("3f", "3 of 8, drawn per forward"),
            ("3t", "3 of 8, drawn per token")]
LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4]


def compact(out, paths):
    runs = []
    for p in paths:
        for r in json.load(open(p))["runs"]:
            if r["scores"] != "abs":
                continue
            keep = {k: r[k] for k in ("name", "kind", "k", "j", "tau", "lam", "variant",
                                      "val", "val_sto", "last_step")}
            keep["stopped"] = r["stopped"] is not None
            keep["diag"] = r["diag"] if r["last_step"] > STOP else []
            if keep["diag"]:
                keep["diag"] = [{k: x[k] for k in ("step", "train/ce",
                                                    "bottleneck/gumbel_exchange_frac",
                                                    "gumbel/grad_split/ratio") if k in x}
                                for x in keep["diag"]]
            runs.append(keep)
    json.dump({"runs": runs}, open(out, "w"))
    print(f"{len(runs)} runs -> {out}")


def val_at(r, s):
    for a, v in r["val"]:
        if a == s:
            return v
    return math.nan


def plot(path, figdir):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    os.makedirs(figdir, exist_ok=True)
    runs = json.load(open(path))["runs"]
    hard = {r["k"]: r for r in runs if r["kind"] == "hard"}
    hk = sorted(hard)
    final = {(r["variant"], r["k"], r["j"]): r for r in runs
             if r["kind"] == "gum" and r["last_step"] >= 20000}
    plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.titlesize": 10.5})

    def ramp(cmap, i, n):
        return cmap(0.4 + 0.55 * (i / max(1, n - 1)))

    def curve(ax, r, lo=0, **kw):
        pts = [(s, v) for s, v in r["val"] if s >= lo]
        ax.plot([s for s, _ in pts], [v for _, v in pts], **kw)

    # ---- hard ladder ------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, lo in zip(axes, (0, 8000)):
        for i, k in enumerate(hk):
            curve(ax, hard[k], lo, color=ramp(cm.Reds, i, len(hk)), lw=1.6,
                  label=f"hard Top-{k}")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(1.4, 2.0)
    axes[0].set_xlim(0, 20000)
    axes[1].set_xlim(8000, 20000)
    axes[1].set_ylim(1.45, 1.85)
    axes[0].set_ylabel("validation CE (nats)")
    axes[0].set_title("whole run")
    axes[1].set_title("steps 8000-20000")
    axes[1].legend()
    fig.suptitle("Hard Top-K, abs-TopK selection, code carried, no output norm, constant LR")
    fig.tight_layout()
    fig.savefig(f"{figdir}/hard_ladder.png", dpi=150)
    plt.close(fig)

    # ---- per active count -------------------------------------------------- #
    for K in sorted({k for (_, k, _) in final}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex="row", sharey="row")
        for col, (V, vlabel) in enumerate(VARIANTS):
            js = sorted(j for (v, k, j) in final if v == V and k == K)
            for row, lo in enumerate((0, 10000)):
                ax = axes[row, col]
                for i, k in enumerate(hk):
                    curve(ax, hard[k], lo, color=ramp(cm.Reds, i, len(hk)), ls="--", lw=1.1,
                          label=f"hard Top-{k}")
                for i, j in enumerate(js):
                    r = final[(V, K, j)]
                    curve(ax, r, lo, color=ramp(cm.Blues, i, len(js)), lw=1.6,
                          label=f"K={K}, J={j} (tau {r['tau']:g}, lambda {r['lam']:g})")
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title(vlabel)
                    ax.set_ylim(1.4, 2.0)
                    ax.set_xlim(0, 20000)
                else:
                    ax.set_xlim(10000, 20000)
                    # from hard Top-2K at the end to hard Top-K at step 10000
                    ax.set_ylim(val_at(hard[min(hk[-1], 2 * K)], 20000) - 0.01,
                                val_at(hard[K], 10000) + 0.015)
                    ax.set_xlabel("training step")
            axes[1, col].legend(fontsize=7, loc="upper right")
        axes[0, 0].set_ylabel("validation CE (nats)")
        axes[1, 0].set_ylabel("validation CE (nats)")
        fig.suptitle(f"Active count K = {K}: every candidate pool K+J against all hard Top-K'")
        fig.tight_layout()
        fig.savefig(f"{figdir}/active_k{K}.png", dpi=150)
        plt.close(fig)

    # ---- per pool ----------------------------------------------------------- #
    from matplotlib.lines import Line2D
    pools = sorted({k + j for (_, k, j) in final})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    styles = {"": "-", "3f": "--", "3t": ":"}
    kcols = {16: "tab:blue", 32: "tab:green", 64: "tab:purple"}
    for ax, P in zip(axes.flat, pools):
        for i, k in enumerate(hk):
            if k == P:
                curve(ax, hard[k], 10000, color="black", ls="--", lw=1.8)
            else:
                curve(ax, hard[k], 10000, color=ramp(cm.Reds, i, len(hk)), ls="--", lw=0.9)
                pts = [v for s, v in hard[k]["val"] if s == 20000]
                if pts:
                    ax.text(20100, pts[0], f"Top-{k}", fontsize=7, va="center",
                            color=ramp(cm.Reds, i, len(hk)), clip_on=True)
        for (V, k, j), r in sorted(final.items()):
            if k + j == P:
                curve(ax, r, 10000, color=kcols.get(k, "gray"), ls=styles[V], lw=1.6)
        ax.set_xlim(10000, 21200)
        ax.set_ylim(val_at(hard[P], 20000) - 0.02 if P in hard else 1.45, 1.83)
        ax.set_title(f"pool K+J = {P}")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation CE (nats)")
    handles = ([Line2D([], [], color="black", ls="--", lw=1.8, label="hard Top-(K+J)"),
                Line2D([], [], color=cm.Reds(0.7), ls="--", lw=0.9, label="other hard Top-K'")]
               + [Line2D([], [], color=c, lw=2, label=f"Gumbel, K={k}") for k, c in kcols.items()]
               + [Line2D([], [], color="gray", ls=styles[v], lw=1.6, label=lab)
                  for v, lab in VARIANTS])
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Candidate pool K+J: Gumbel cells against hard Top-(K+J) and the other hard "
                 "runs, steps 10000-20000")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{figdir}/pools.png", dpi=150)
    plt.close(fig)

    # ---- search ------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    tcols = {0.2: "tab:blue", 0.5: "tab:orange", 1.2: "tab:red"}
    for ax, (V, vlabel) in zip(axes, VARIANTS):
        sr = [r for r in runs if r["kind"] == "gum" and r["variant"] == V and r["stopped"]]
        for tau in (0.2, 0.5, 1.2):
            means, los, his = [], [], []
            for lam in LAMBDAS:
                m = [val_at(r, STOP) - val_at(hard[r["k"]], STOP) for r in sr
                     if r["tau"] == tau and r["lam"] == lam and math.isfinite(val_at(r, STOP))]
                means.append(sum(m) / len(m) if m else math.nan)
                los.append(min(m) if m else math.nan)
                his.append(max(m) if m else math.nan)
            xs = range(len(LAMBDAS))
            ax.plot(xs, means, "o-", color=tcols[tau], label=f"tau {tau:g}")
            ax.fill_between(xs, los, his, color=tcols[tau], alpha=0.15)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_xticks(range(len(LAMBDAS)), ["0", "1e-5", "3e-5", "1e-4"])
        ax.set_xlabel("lambda")
        ax.set_title(vlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val CE at 2500 minus hard Top-K")
    axes[0].legend()
    fig.suptitle("(tau, lambda) search: margin over hard Top-K at step 2500, mean over the "
                 "nine (K, J) cells (band: min-max)")
    fig.tight_layout()
    fig.savefig(f"{figdir}/search.png", dpi=150)
    plt.close(fig)

    # ---- exchange fraction ------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cells = sorted((k, j) for (v, k, j) in final if v == "")
    for i, (k, j) in enumerate(cells):
        r = final[("", k, j)]
        d = [x for x in r["diag"] if "bottleneck/gumbel_exchange_frac" in x]
        ax.plot([x["step"] for x in d], [x["bottleneck/gumbel_exchange_frac"] for x in d],
                color=kcols[k], alpha=0.45 + 0.55 * (j / 240), lw=1.3,
                label=f"K={k}, J={j}, tau {r['tau']:g}")
    ax.set_xlabel("training step")
    ax.set_ylabel("share of drawn support outside TopK")
    ax.set_title("Exchange fraction, all-bottleneck variant, selected 20000-step runs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{figdir}/exchange.png", dpi=150)
    plt.close(fig)

    # ---- tables ------------------------------------------------------------- #
    print("% final val/ce at 20000 (rows: variant, K; columns: K+J = 32, 64, 128, 256)")
    for V, vlabel in VARIANTS:
        for K in (16, 32, 64):
            row = []
            for P in (32, 64, 128, 256):
                r = final.get((V, K, P - K))
                row.append(f"{val_at(r, 20000):.4f}" if r else "--")
            print(f"{vlabel} & {K} & " + " & ".join(row)
                  + f" & {val_at(hard[K], 20000):.4f} \\\\")
    print("hard Top-(K+J) & & " + " & ".join(f"{val_at(hard[P], 20000):.4f}"
                                            for P in (32, 64, 128, 256)) + " \\\\")
    print("% selected (tau, lambda) per cell and variant")
    for V, vlabel in VARIANTS:
        print(vlabel, ", ".join(f"({k},{j}): {r['tau']:g}/{r['lam']:g}"
                                for (v, k, j), r in sorted(final.items()) if v == V))


if __name__ == "__main__":
    if sys.argv[1] == "compact":
        compact(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
