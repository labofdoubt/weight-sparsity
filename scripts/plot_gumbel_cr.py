"""Gumbel Top-(K+J) against hard Top-K under the code residual: data, figures, tables.

  python scripts/plot_gumbel_cr.py extract OUT.json RUNS_DIR [RUNS_DIR ...]
  python scripts/plot_gumbel_cr.py plot OUT.json FIGDIR

``extract`` reads every ``gcr_*`` run directory (config.json, metrics.jsonl,
stopped.json / diverged.json / summary.json) and writes one compact JSON: per
run the score type (abs: support ranked by |a|; sep: by the separately
computed s = W_s h + b_s), K, J, tau, lambda, the deterministic validation
curve, the paired sampled one, and every 100th logged training diagnostic.

``plot`` writes (vertical axes capped at 2.0 nats, as in
docs/kj-vs-hard-topk-code-residual.tex):

  hard_ladders.png     hard Top-K, both score types (abs dashed red ramp,
                       separate solid blue ramp; lightness = K)
  active_k<K>.png      one active count K: every 20000-step Gumbel cell with
                       that K (solid, blues light-to-dark with J) against all
                       hard Top-K' of the same score type (dashed, reds light-
                       to-dark with K'); left abs-TopK scores, right separate
  pool_k<K'>.png       one pool K' = K+J: every Gumbel cell with that pool
                       (solid, blues light-to-dark with K) against all hard
                       Top-K'' (dashed reds), hard Top-K' in black
  search_<scores>.png  the (tau, lambda) search: per searched (K, J), val/ce at
                       the stop step minus hard Top-K at the same step
  search_curves_<scores>.png  the search runs' validation curves to step 2500
  diag.png             the 20000-step Gumbel runs: policy-term / CE gradient
                       ratio and the exchange fraction over training

and prints the tables (final val/ce, search margins) as LaTeX rows.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

STOP = 2500


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #

def read_run(d: str) -> dict | None:
    name = os.path.basename(d.rstrip("/"))
    cfg_path = os.path.join(d, "config.json")
    met = os.path.join(d, "metrics.jsonl")
    if not (os.path.exists(cfg_path) and os.path.exists(met)):
        return None
    cfg = json.load(open(cfg_path))
    ab = cfg["activation_bottleneck"]
    run = {"name": name, "dir": d,
           "scores": "abs" if ab["selection_mode"] == "abs_topk" else "sep",
           "kind": "gum" if ab.get("gumbel_policy") else "hard",
           "k": int(ab["k"]), "j": int(ab["j"]),
           "tau": float(ab.get("gumbel_temperature", 0.0)),
           "lam": float(ab.get("gumbel_policy_coef", 0.0)),
           # which bottlenecks sample: "" all 8, "3f" 3 of 8 per forward, "3t" per token
           "variant": ("" if not int(ab.get("gumbel_gates_per_forward", 0) or 0) else
                       "3f" if ab.get("gumbel_gates_draw", "forward") == "forward" else "3t"),
           "val": [], "val_sto": [], "diag": []}
    diag_keys = ("train/ce", "train/grad_norm", "gumbel/grad_split/ratio",
                 "gumbel/grad_split/policy_norm", "gumbel/grad_split/ce_norm",
                 "gumbel/grad_split/ce_norm_scorer", "gumbel/grad_split/policy_norm_scorer",
                 "bottleneck/gumbel_exchange_frac", "bottleneck/gumbel_margin",
                 "bottleneck/gumbel_logit_std", "gumbel/advantage_rms")
    seen_val = {}
    with open(met) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = r.get("step")
            if "val/ce" in r and s is not None:
                seen_val[s] = (r["val/ce"], r.get("val_stochastic/ce"))
            elif s is not None and s % 200 == 0 and "train/ce" in r:
                run["diag"].append({"step": s, **{k: r[k] for k in diag_keys if k in r}})
    for s in sorted(seen_val):
        v, vs = seen_val[s]
        run["val"].append([s, v])
        if vs is not None:
            run["val_sto"].append([s, vs])
    for tag in ("stopped", "diverged", "summary"):
        p = os.path.join(d, f"{tag}.json")
        run[tag] = json.load(open(p)) if os.path.exists(p) else None
    run["last_step"] = run["val"][-1][0] if run["val"] else 0
    if run["last_step"] <= STOP:
        run["diag"] = run["diag"][::5]   # search runs: every 1000 steps is enough
    return run


def extract(out: str, dirs) -> None:
    runs = []
    for root in dirs:
        for d in sorted(glob.glob(os.path.join(root, "gcr_*"))):
            if os.path.isdir(d):
                r = read_run(d)
                if r is not None:
                    runs.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"runs": runs}, f)
    print(f"{len(runs)} runs -> {out}")


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #

def val_at(run, step):
    for s, v in run["val"]:
        if s == step:
            return v
    return math.nan


def final(run):
    return run["val"][-1][1] if run["val"] and run["last_step"] >= 20000 else math.nan


def plot(data_path: str, figdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    runs = json.load(open(data_path))["runs"]
    os.makedirs(figdir, exist_ok=True)
    hard = {(r["scores"], r["k"]): r for r in runs if r["kind"] == "hard"}
    gum = [r for r in runs if r["kind"] == "gum"]
    long_gum = {(r["scores"], r["k"], r["j"]): r for r in gum if r["last_step"] > STOP}
    hard_ks = sorted({k for (_, k) in hard})
    label = {"abs": "abs-TopK scores", "sep": "separate scores"}

    def ramp(cmap, i, n):
        return cmap(0.35 + 0.6 * (i / max(1, n - 1)))

    def curve(ax, run, **kw):
        xs = [s for s, _ in run["val"]]
        ys = [v for _, v in run["val"]]
        ax.plot(xs, ys, **kw)

    def style(ax, title):
        ax.set_ylim(1.35, 2.0)
        ax.set_xlim(0, 20000)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=10)

    # hard ladders
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sc, cmap, ls in (("abs", cm.Reds, "--"), ("sep", cm.Blues, "-")):
        ks = [k for k in hard_ks if (sc, k) in hard]
        for i, k in enumerate(ks):
            curve(ax, hard[(sc, k)], color=ramp(cmap, i, len(ks)), ls=ls,
                  label=f"{label[sc]}, hard Top-{k}")
    style(ax, "Hard Top-K, code carried, no output norm")
    ax.set_ylabel("validation CE (nats)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "hard_ladders.png"), dpi=150)
    plt.close(fig)

    # per active count
    for K in sorted({k for (_, k, _) in long_gum}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, sc in zip(axes, ("abs", "sep")):
            ks = [k for k in hard_ks if (sc, k) in hard]
            for i, k in enumerate(ks):
                curve(ax, hard[(sc, k)], color=ramp(cm.Reds, i, len(ks)), ls="--",
                      lw=1.2, label=f"hard Top-{k}")
            js = sorted(j for (s, k, j) in long_gum if s == sc and k == K)
            for i, j in enumerate(js):
                r = long_gum[(sc, K, j)]
                curve(ax, r, color=ramp(cm.Blues, i, len(js)), lw=1.6,
                      label=f"Gumbel K={K}, J={j} (tau {r['tau']:g}, lambda {r['lam']:g})")
            style(ax, label[sc])
            ax.legend(fontsize=7)
        axes[0].set_ylabel("validation CE (nats)")
        fig.suptitle(f"Active count K = {K}: every candidate window against all hard Top-K'",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"active_k{K}.png"), dpi=150)
        plt.close(fig)

    # per pool
    for P in sorted({k + j for (_, k, j) in long_gum}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, sc in zip(axes, ("abs", "sep")):
            ks = [k for k in hard_ks if (sc, k) in hard]
            for i, k in enumerate(ks):
                if k == P:
                    curve(ax, hard[(sc, k)], color="black", ls="--", lw=1.6,
                          label=f"hard Top-{k} (= pool)")
                else:
                    curve(ax, hard[(sc, k)], color=ramp(cm.Reds, i, len(ks)), ls="--",
                          lw=1.0, label=f"hard Top-{k}")
            cells = sorted((k, j) for (s, k, j) in long_gum if s == sc and k + j == P)
            for i, (k, j) in enumerate(cells):
                r = long_gum[(sc, k, j)]
                curve(ax, r, color=ramp(cm.Blues, i, len(cells)), lw=1.6,
                      label=f"Gumbel K={k}, J={j}")
            style(ax, label[sc])
            ax.legend(fontsize=7)
        axes[0].set_ylabel("validation CE (nats)")
        fig.suptitle(f"Candidate pool K+J = {P}: Gumbel cells against all hard Top-K",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"pool_k{P}.png"), dpi=150)
        plt.close(fig)

    # search heatmaps and tables
    import numpy as np
    for sc in ("abs", "sep"):
        # searched cells: their runs were stopped at STOP (stopped.json stays
        # in the directory of the one resumed to 20000)
        cells = sorted({(r["k"], r["j"]) for r in gum
                        if r["scores"] == sc and r["stopped"] is not None})
        if not cells:
            continue
        taus = sorted({r["tau"] for r in gum if r["scores"] == sc})
        lams = sorted({r["lam"] for r in gum if r["scores"] == sc})
        n = len(cells)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.2))
        axes = np.atleast_1d(axes)
        print(f"\n% search margins at step {STOP}, {label[sc]} (val/ce minus hard Top-K)")
        for ax, (k, j) in zip(axes, cells):
            ref = val_at(hard[(sc, k)], STOP) if (sc, k) in hard else math.nan
            M = np.full((len(taus), len(lams)), np.nan)
            for r in gum:
                if r["scores"] == sc and (r["k"], r["j"]) == (k, j):
                    M[taus.index(r["tau"]), lams.index(r["lam"])] = val_at(r, STOP) - ref
            im = ax.imshow(M, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
            for a in range(len(taus)):
                for b in range(len(lams)):
                    if np.isfinite(M[a, b]):
                        ax.text(b, a, f"{M[a, b]:+.3f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(lams)), [f"{l:g}" for l in lams], fontsize=7)
            ax.set_yticks(range(len(taus)), [f"{t:g}" for t in taus], fontsize=7)
            ax.set_xlabel("lambda", fontsize=8)
            ax.set_title(f"K={k}, J={j}", fontsize=9)
            print(f"% K={k} J={j} ref hard Top-{k} = {ref:.4f}")
            for a, t in enumerate(taus):
                print(f"{k} & {j} & {t:g} & " + " & ".join(
                    f"{M[a, b]:+.3f}" if np.isfinite(M[a, b]) else "--"
                    for b in range(len(lams))) + r" \\")
        axes[0].set_ylabel("tau", fontsize=8)
        fig.colorbar(im, ax=list(axes), shrink=0.8, label=f"val CE at {STOP} minus hard Top-K")
        fig.suptitle(f"(tau, lambda) search, {label[sc]}", fontsize=10)
        fig.savefig(os.path.join(figdir, f"search_{sc}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (k, j) in zip(axes, cells):
            if (sc, k) in hard:
                h = hard[(sc, k)]
                ax.plot([s for s, _ in h["val"] if s <= STOP], [v for s, v in h["val"] if s <= STOP],
                        "k--", lw=1.2, label=f"hard Top-{k}")
            for r in gum:
                if r["scores"] == sc and (r["k"], r["j"]) == (k, j):
                    ti, li = taus.index(r["tau"]), lams.index(r["lam"])
                    col = cm.viridis(li / max(1, len(lams) - 1))
                    ls = ["-", "--", ":"][ti % 3]
                    pts = [(s, v) for s, v in r["val"] if s <= STOP]
                    ax.plot([s for s, _ in pts], [v for _, v in pts], color=col, ls=ls, lw=1,
                            label=f"tau {r['tau']:g}, lambda {r['lam']:g}")
            ax.set_ylim(1.6, 3.0)
            ax.grid(alpha=0.3)
            ax.set_title(f"K={k}, J={j}", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
        axes[0].set_ylabel("validation CE (nats)")
        axes[-1].legend(fontsize=5, ncol=1, loc="upper right")
        fig.suptitle(f"(tau, lambda) search runs to step {STOP}, {label[sc]}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"search_curves_{sc}.png"), dpi=150)
        plt.close(fig)

    # final tables
    for sc in ("abs", "sep"):
        print(f"\n% final val/ce at 20000, {label[sc]}: rows K, columns K+J")
        pools = sorted({k + j for (s, k, j) in long_gum if s == sc})
        for K in sorted({k for (s, k, _) in long_gum if s == sc}):
            row = []
            for P in pools:
                r = long_gum.get((sc, K, P - K))
                row.append(f"{final(r):.4f}" if r and math.isfinite(final(r)) else "--")
            print(f"{K} & " + " & ".join(row) + r" \\")
        print("hard Top-K' & " + " & ".join(
            f"{final(hard[(sc, P)]):.4f}" if (sc, P) in hard and math.isfinite(final(hard[(sc, P)]))
            else "--" for P in pools) + r" \\")
        print("% hard ladder: " + ", ".join(
            f"K={k}: {final(hard[(sc, k)]):.4f}" for k in hard_ks if (sc, k) in hard))
        for (s, k, j), r in sorted(long_gum.items()):
            if s == sc:
                print(f"%   {r['name']}: final {final(r):.4f} (step {r['last_step']})")

    # diagnostics of the 20000-step Gumbel runs
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, sc in enumerate(("abs", "sep")):
        items = sorted((k, j) for (s, k, j) in long_gum if s == sc)
        for i, (k, j) in enumerate(items):
            r = long_gum[(sc, k, j)]
            c = cm.tab10(i % 10)
            d = [x for x in r["diag"] if "gumbel/grad_split/ratio" in x]
            axes[0, col].plot([x["step"] for x in d], [x["gumbel/grad_split/ratio"] for x in d],
                              color=c, lw=0.8, label=f"K={k}, J={j}")
            d = [x for x in r["diag"] if "bottleneck/gumbel_exchange_frac" in x]
            axes[1, col].plot([x["step"] for x in d],
                              [x["bottleneck/gumbel_exchange_frac"] for x in d], color=c, lw=0.8)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(f"{label[sc]}: policy-term / CE gradient norm", fontsize=9)
        axes[1, col].set_title(f"{label[sc]}: exchange fraction", fontsize=9)
        axes[1, col].set_xlabel("training step")
        axes[0, col].legend(fontsize=6, ncol=2)
        for a in axes[:, col]:
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "diag.png"), dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
