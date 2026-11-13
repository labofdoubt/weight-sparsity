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
  lpos.png            the lambda = 1e-4 runs continued from the search to 20000 (tau
                      0.5 and 1.2) and the lambda = 3e-5 runs that replaced the ones
                      the guard stopped: margin over hard Top-K at the same step
                      against the selected lambda = 0 runs and hard Top-2K; rows K,
                      columns variants, symmetric-log axis; x = collapsed
  lpos_late.png       the lambda > 0 runs that did not collapse, steps 10000-20000,
                      linear axis, one panel per K (line style: variant)
  lpos_collapse.png   the lambda > 0 runs that trained and then collapsed: training CE,
                      share of the drawn support outside TopK, policy/CE gradient ratio

The selected runs are the lambda = 0 runs that reached step 20000 (lambda = 0 was
selected in every cell); every other run past step 2500 is a lambda > 0 run.
A lambda > 0 run counts as collapsed when the divergence guard stopped it, or when
the share of its drawn support outside TopK, averaged over its last 500 logged
steps, is below 0.02 (the sampling has become deterministic).
"""
import json
import math
import sys

STOP = 2500
VARIANTS = [("", "all 8 bottlenecks sample"), ("3f", "3 of 8, drawn per forward"),
            ("3t", "3 of 8, drawn per token")]
VLABEL_SHORT = {"": "all 8", "3f": "3 of 8 per forward", "3t": "3 of 8 per token"}
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
            # a run past the search that the divergence guard stopped
            dv = r.get("diverged")
            keep["end"] = (f"diverged at step {dv['step']} ({dv['reason']})"
                           if dv and r["last_step"] > STOP else "")
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
    # the selected 20000-step runs (lambda = 0 in every cell) ...
    final = {(r["variant"], r["k"], r["j"]): r for r in runs
             if r["kind"] == "gum" and r["last_step"] >= 20000 and r["lam"] == 0}
    # ... and the lambda > 0 runs continued past the search (lambda 1e-4 at two tau;
    # lambda 3e-5 where the lambda 1e-4 run collapsed)
    lpos = {(r["variant"], r["k"], r["j"], r["tau"], r["lam"]): r for r in runs
            if r["kind"] == "gum" and r["last_step"] > STOP and r["lam"] > 0}
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

    # ---- lambda > 0 continued to 20000 ------------------------------------- #
    if lpos:
        lpos_figure(plt, runs, hard, final, lpos, figdir)

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


def margin(r, h, lo=0):
    """val/ce of run r minus hard run h at the steps both logged (>= lo)."""
    hv = dict((s, v) for s, v in h["val"])
    return [(s, v - hv[s]) for s, v in r["val"] if s in hv and s >= lo]


def exchange_end(r, span=500):
    """Mean share of the drawn support outside TopK over the last `span` steps."""
    d = [x["bottleneck/gumbel_exchange_frac"] for x in r["diag"]
         if x["step"] > r["last_step"] - span and "bottleneck/gumbel_exchange_frac" in x]
    return sum(d) / len(d) if d else math.nan


def exchange_at(r, s, span=500):
    """Mean share of the drawn support outside TopK over steps s..s+span."""
    d = [x["bottleneck/gumbel_exchange_frac"] for x in r["diag"]
         if s <= x["step"] <= s + span and "bottleneck/gumbel_exchange_frac" in x]
    return sum(d) / len(d) if d else math.nan


def collapsed(r):
    """Stopped by the divergence guard, or sampling deterministic at the end."""
    return "diverged" in r.get("end", "") or exchange_end(r) < 0.02


def slope(r, h, lo, hi=20000):
    """Least-squares slope of the margin over hard run h, steps lo..hi, per 10000 steps."""
    m = [(s, d) for s, d in margin(r, h, lo) if s <= hi]
    mx = sum(s for s, _ in m) / len(m)
    my = sum(d for _, d in m) / len(m)
    return 1e4 * (sum((s - mx) * (d - my) for s, d in m) / sum((s - mx) ** 2 for s, _ in m))


def lpos_figure(plt, runs, hard, final, lpos, figdir):
    """lpos.png: rows K = 16, 32, 64, columns the variants.  Each panel: the
    margin over hard Top-K at the same step for the lambda = 1e-4 runs (tau 0.5
    orange, tau 1.2 red; one line per J, lighter for smaller J), the lambda = 3e-5
    replacements (dotted), the selected lambda = 0 runs (blue) and hard Top-2K
    (black dashed), symmetric-log axis; x marks the end of a collapsed run.
    lpos_late.png: the runs that did not collapse, steps 10000-20000, linear."""
    from matplotlib.lines import Line2D
    Ks = (16, 32, 64)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True)
    tcol = {0.5: "tab:orange", 1.2: "tab:red"}
    lstyle = {1e-4: "-", 3e-5: ":"}
    for row, K in enumerate(Ks):
        for col, (V, vlabel) in enumerate(VARIANTS):
            ax = axes[row, col]
            js = sorted({j for (v, k, j, _, _) in lpos if v == V and k == K}
                        | {j for (v, k, j) in final if v == V and k == K})
            for i, j in enumerate(js):
                a = 0.45 + 0.55 * (i / max(1, len(js) - 1))
                for tau, c in tcol.items():
                    for lam, ls in lstyle.items():
                        r = lpos.get((V, K, j, tau, lam))
                        if r:
                            m = margin(r, hard[K])
                            ax.plot([s for s, _ in m], [d for _, d in m], color=c, alpha=a,
                                    ls=ls, lw=1.3)
                            if collapsed(r):
                                ax.plot(m[-1][0], m[-1][1], "x", color=c, ms=7, mew=2)
                r = final.get((V, K, j))
                if r:
                    m = margin(r, hard[K])
                    ax.plot([s for s, _ in m], [d for _, d in m], color="tab:blue", alpha=a,
                            lw=1.3)
            if 2 * K in hard:
                m = margin(hard[2 * K], hard[K])
                ax.plot([s for s, _ in m], [d for _, d in m], color="black", ls="--", lw=1.2)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_yscale("symlog", linthresh=0.05, linscale=1.5)
            ax.set_ylim(-0.15, 6.0)
            ax.set_xlim(0, 20600)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(vlabel)
            if row == 2:
                ax.set_xlabel("training step")
            if col == 0:
                ax.set_ylabel(f"K = {K}: val CE minus hard Top-{K}")
    handles = [Line2D([], [], color="tab:orange", lw=2, label="lambda 1e-4, tau 0.5"),
               Line2D([], [], color="tab:red", lw=2, label="lambda 1e-4, tau 1.2"),
               Line2D([], [], color="gray", lw=2, ls=":",
                      label="lambda 3e-5 (where the guard stopped 1e-4)"),
               Line2D([], [], color="gray", marker="x", ls="", mew=2, label="collapsed"),
               Line2D([], [], color="tab:blue", lw=2, label="lambda 0 (selected tau)"),
               Line2D([], [], color="black", ls="--", lw=1.2, label="hard Top-2K"),
               Line2D([], [], color="gray", lw=2, alpha=0.5, label="lighter: smaller J")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.003))
    fig.suptitle("Margin over hard Top-K at the same step: policy term on (lambda > 0) "
                 "against off (lambda 0), symmetric-log axis")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{figdir}/lpos.png", dpi=150)
    plt.close(fig)

    # ---- the runs that did not collapse, late training, linear axis -------- #
    ok = {key: r for key, r in lpos.items() if not collapsed(r)}
    vstyle = {"": "-", "3f": "--", "3t": ":"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, K in zip(axes, Ks):
        for (V, k, j, tau, lam), r in sorted(ok.items()):
            if k != K:
                continue
            c = "tab:purple" if lam == 3e-5 else tcol[tau]
            m = margin(r, hard[K], 10000)
            ax.plot([s for s, _ in m], [d for _, d in m], color=c, ls=vstyle[V], lw=1.2)
        for (V, k, j), r in sorted(final.items()):
            if k == K:
                m = margin(r, hard[K], 10000)
                ax.plot([s for s, _ in m], [d for _, d in m], color="tab:blue", ls=vstyle[V],
                        lw=1.0, alpha=0.8)
        if 2 * K in hard:
            m = margin(hard[2 * K], hard[K], 10000)
            ax.plot([s for s, _ in m], [d for _, d in m], color="black", ls="--", lw=1.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlim(10000, 20000)
        ax.set_ylim(-0.11, 0.30)
        ax.set_title(f"K = {K}")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val CE minus hard Top-K (nats)")
    handles = [Line2D([], [], color="tab:orange", lw=2, label="lambda 1e-4, tau 0.5"),
               Line2D([], [], color="tab:red", lw=2, label="lambda 1e-4, tau 1.2"),
               Line2D([], [], color="tab:purple", lw=2, label="lambda 3e-5"),
               Line2D([], [], color="tab:blue", lw=2, label="lambda 0 (selected)"),
               Line2D([], [], color="black", ls="--", lw=1.4, label="hard Top-2K")]
    handles += [Line2D([], [], color="gray", ls=vstyle[v], lw=1.6, label=lab)
                for v, lab in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Runs with lambda > 0 that did not collapse, steps 10000-20000: margin over "
                 "hard Top-K at the same step (cut at 0.30)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{figdir}/lpos_late.png", dpi=150)
    plt.close(fig)

    # ---- the runs that trained and then collapsed --------------------------- #
    late = sorted(((V, k, j, tau, lam), r) for (V, k, j, tau, lam), r in lpos.items()
                  if collapsed(r) and exchange_at(r, STOP) >= 0.025)
    if late:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
        keys = ("train/ce", "bottleneck/gumbel_exchange_frac", "gumbel/grad_split/ratio")
        cols = plt.get_cmap("tab10")
        for i, ((V, k, j, tau, lam), r) in enumerate(late):
            lab = (f"{VLABEL_SHORT[V]}, ({k},{j}), tau {tau:g}, lambda {lam:g}")
            for ax, key in zip(axes, keys):
                d = [(x["step"], x[key]) for x in r["diag"] if key in x and x["step"] >= STOP]
                ax.plot([s for s, _ in d], [v for _, v in d], color=cols(i), lw=1.0,
                        label=lab)
        axes[0].set_ylabel("training CE (nats)")
        axes[1].set_ylabel("share of drawn support outside TopK")
        axes[2].set_ylabel("policy / CE gradient norm")
        axes[2].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("training step")
            ax.grid(alpha=0.3)
            ax.set_xlim(STOP, 20000)
        axes[0].legend(fontsize=7)
        fig.suptitle("lambda > 0 runs that trained and then collapsed (diagnostics every 200 steps)")
        fig.tight_layout()
        fig.savefig(f"{figdir}/lpos_collapse.png", dpi=150)
        plt.close(fig)

    # ---- tables ------------------------------------------------------------- #
    steps = (2500, 5000, 10000, 15000, 20000)
    groups = [("lambda 1e-4, tau 0.5", lambda t, l: l == 1e-4 and t == 0.5),
              ("lambda 1e-4, tau 1.2", lambda t, l: l == 1e-4 and t == 1.2),
              ("lambda 3e-5", lambda t, l: l == 3e-5)]
    print("% lambda > 0, runs that did not collapse: margin over hard Top-K, "
          "mean [min, max] over runs (n)")
    for V, vlabel in VARIANTS:
        for glabel, sel in groups:
            cells = [r for (v, k, j, t, l), r in ok.items() if v == V and sel(t, l)]
            out = []
            for s in steps:
                d = [val_at(r, s) - val_at(hard[r["k"]], s) for r in cells]
                d = [x for x in d if math.isfinite(x)]
                out.append(f"{sum(d) / len(d):+.3f} [{min(d):+.3f}, {max(d):+.3f}] (n={len(d)})"
                           if d else "--")
            print(f"{vlabel}, {glabel}: " + " | ".join(f"{s}: {o}" for s, o in zip(steps, out)))
        d = {s: [val_at(r, s) - val_at(hard[r["k"]], s) for (v, _, _), r in final.items()
                 if v == V] for s in steps}
        print(f"{vlabel}, lambda 0 selected: " + " | ".join(
            f"{s}: {sum(x) / len(x):+.3f} [{min(x):+.3f}, {max(x):+.3f}]"
            for s, x in d.items() if x))
    print("% collapsed lambda > 0 runs: exchange at 2500 and at the end; val at 2500, min "
          "after 2500, at the end")
    for key, r in sorted(lpos.items()):
        if collapsed(r):
            post = [(s, v) for s, v in r["val"] if s >= STOP]
            smin, vmin = min(post, key=lambda x: x[1])
            x25 = [x["bottleneck/gumbel_exchange_frac"] for x in r["diag"]
                   if STOP <= x["step"] <= STOP + 500 and "bottleneck/gumbel_exchange_frac" in x]
            print(f"  {r['name']}: exch {sum(x25) / len(x25):.3f} -> {exchange_end(r):.3f}; "
                  f"val {val_at(r, STOP):.3f}, min {vmin:.3f}@{smin}, end {post[-1][1]:.3f}"
                  f"@{post[-1][0]}; {r['end'] or 'not stopped by the guard'}")
    print("% the runs that did not collapse, closest to hard Top-K at 20000: margin, slope of "
          "the margin per 10000 steps over 10000-20000 and 15000-20000")
    rows = sorted((val_at(r, 20000) - val_at(hard[r["k"]], 20000), r) for r in ok.values()
                  if math.isfinite(val_at(r, 20000)))
    for m, r in rows[:8]:
        print(f"  {r['name']}: {m:+.4f}, slope {slope(r, hard[r['k']], 10000):+.4f}, "
              f"{slope(r, hard[r['k']], 15000):+.4f}")
    for lo in (10000, 15000):
        b = sorted(slope(r, hard[r["k"]], lo) for _, r in rows)
        print(f"  slope over {lo}-20000, all {len(b)} runs: median {b[len(b) // 2]:+.4f}, "
              f"{sum(x < 0 for x in b)} negative")
    print("% per K: closest lambda > 0 run at 20000, lambda 0 selected range, hard Top-2K")
    for K in Ks:
        mk = [m for m, r in rows if r["k"] == K]
        m0 = [val_at(r, 20000) - val_at(hard[K], 20000) for (v, k, j), r in final.items()
              if k == K]
        print(f"  K={K}: lambda>0 min {min(mk):+.4f} (n={len(mk)}); lambda 0 "
              f"[{min(m0):+.4f}, {max(m0):+.4f}]; hard Top-{2 * K} "
              f"{val_at(hard[2 * K], 20000) - val_at(hard[K], 20000):+.4f}")
    print(f"% counts: {len(lpos)} lambda > 0 runs past {STOP}, "
          f"{sum(collapsed(r) for r in lpos.values())} collapsed "
          f"({sum('diverged' in r['end'] for r in lpos.values())} stopped by the guard)")

if __name__ == "__main__":
    if sys.argv[1] == "compact":
        compact(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
