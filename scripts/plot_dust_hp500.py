"""hp500: the DUST sigma x dust_lr_scale grid over the first 500 steps -- figures and tables.

Reads the run directories of the grid (metrics.jsonl, config.json, summary.json;
runs named *_hp500_<enc|bp>_sg<sigma>_lr<scale>) from one or more roots -- the
local TensorBoard mirror, or a copy pulled from Drive -- and writes, per
dust_target, one figure with four panels (one per sigma), each showing the
validation CE over steps for every dust_lr_scale (one hue, light to dark with
the multiplier), with the hard Top-32 baseline run of the grid (the same
model without DUST steps, validated every 50 steps) as a dashed black curve.
Prints Markdown tables of the validation CE at step 500 per cell, the
baseline's value, the original madrid reference's value (``--ref``), and the
run behind every cell.  Cells with a population other than 128 form the sigma
sweeps at dust_lr_scale 1: with MD decoupling, one figure per target with one
curve per sigma (``hp500_p<P>_val_<target>.png``); without it (model.decouple
false, md_init), one two-panel figure per gamma, encoder left and backprop
right, against the hard Top-32 run without decoupling
(``hp500_p<P>_nomd_g<gamma>_val.png``, ``--baseline-nomd``), each with its
table; and ``hp500_hard_md_vs_nomd.png`` compares the two hard runs.

  python scripts/plot_dust_hp500.py --runs-dir <root> [--runs-dir <root2>] \
      --baseline <ny_hp500_hard_k32/metrics.jsonl> \
      --ref <ma_cr_hard_k32_pnorm/metrics.jsonl> --out docs/dust/figures [--ymax 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMAS = [0.03, 0.1, 0.3, 0.9]
GRID_POPULATION = 128
SCALES = [1, 10, 30, 100, 300]
TARGETS = (("encoder", "enc"), ("backprop", "bp"))
STEP_END = 500
BASE_LABEL = "hard Top-32, code carried (baseline: no DUST steps)"


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def val_curve(rows: List[dict], key: str = "val/ce") -> List[Tuple[int, float]]:
    return sorted((int(r["step"]), float(r[key])) for r in rows if key in r)


def load_runs(roots: List[str]) -> Dict[Tuple[str, float, int], dict]:
    runs = {}
    for root in roots:
        for cfg_path in glob.glob(os.path.join(root, "**", "*_hp500_*", "config.json"),
                                  recursive=True):
            run_dir = os.path.dirname(cfg_path)
            cfg = json.load(open(cfg_path))
            ab, tr = cfg["activation_bottleneck"], cfg["train"]
            if ab.get("surrogate_mode") != "dust":
                continue
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.exists(metrics_path):
                continue
            rows = read_jsonl(metrics_path)
            key = (ab["dust_target"], float(ab["dust_sigma"]), int(round(float(ab["dust_lr_scale"]))),
                   int(ab.get("dust_population", 128)), bool(cfg["model"].get("decouple", False)),
                   float(ab.get("dust_gamma", 0.0)))
            runs[key] = {
                "name": tr["run_name"], "dir": run_dir, "val": val_curve(rows),
                "last_step": max((int(r["step"]) for r in rows), default=0),
                "finished": os.path.exists(os.path.join(run_dir, "summary.json")),
                "nonfinite": any(r.get("policy/nonfinite") for r in rows),
            }
    return runs


def value_at(curve: List[Tuple[int, float]], step: int) -> Optional[float]:
    vals = [v for s, v in curve if s == step]
    return vals[-1] if vals else None


def figure(runs, target: str, baseline: List[Tuple[int, float]], out_path: str,
           ymax: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=150, sharex=True, sharey=True)
    ramp = matplotlib.colormaps["Blues"]
    colors = [ramp(0.35 + 0.6 * i / (len(SCALES) - 1)) for i in range(len(SCALES))]
    ymin = None
    for ax, sigma in zip(axes.flat, SIGMAS):
        for color, scale in zip(colors, SCALES):
            run = runs.get((target, sigma, scale, GRID_POPULATION, True, 0.0))
            if run is None or not run["val"]:
                continue
            steps = [s for s, _ in run["val"] if s <= STEP_END]
            vals = [v for s, v in run["val"] if s <= STEP_END]
            ax.plot(steps, vals, color=color, lw=2.0, marker="o", ms=3.5,
                    label=f"dust_lr_scale {scale}")
            finite = [v for v in vals if v == v and v < float("inf")]
            if finite:
                ymin = min(finite) if ymin is None else min(ymin, min(finite))
        if baseline:
            bs = [s for s, _ in baseline if s <= STEP_END]
            bv = [v for s, v in baseline if s <= STEP_END]
            ax.plot(bs, bv, color="black", ls="--", lw=1.8, marker="s", ms=3.0,
                    label=BASE_LABEL)
            ymin = min(bv) if ymin is None else min(ymin, min(bv))
        ax.set_title(f"sigma = {sigma:g}", fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    lo = (ymin - 0.1) if ymin is not None else 0.0
    for ax in axes.flat:
        ax.set_ylim(lo, ymax)
        ax.set_xlim(0, STEP_END)
    for ax in axes[1]:
        ax.set_xlabel("optimizer step")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation CE (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"DUST, dust_target = {target}: validation CE over the first 500 steps "
                 f"(curves above {ymax:g} nats are off scale)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sweep_figure(runs, target: str, population: int, baseline, out_path: str,
                 ymax: float) -> None:
    """One panel: the validation CE of every sigma at dust_lr_scale 1 and the
    given population (one hue, light to dark with sigma), with the baseline."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    ramp = matplotlib.colormaps["Oranges"]
    colors = [ramp(0.4 + 0.55 * i / (len(SIGMAS) - 1)) for i in range(len(SIGMAS))]
    ymin = None
    for color, sigma in zip(colors, SIGMAS):
        run = runs.get((target, sigma, 1, population, True, 0.0))
        if run is None or not run["val"]:
            continue
        steps = [s for s, _ in run["val"] if s <= STEP_END]
        vals = [v for s, v in run["val"] if s <= STEP_END]
        ax.plot(steps, vals, color=color, lw=2.0, marker="o", ms=3.5, label=f"sigma {sigma:g}")
        ymin = min(vals) if ymin is None else min(ymin, min(vals))
    if baseline:
        bs = [s for s, _ in baseline if s <= STEP_END]
        bv = [v for s, v in baseline if s <= STEP_END]
        ax.plot(bs, bv, color="black", ls="--", lw=1.8, marker="s", ms=3.0, label=BASE_LABEL)
        ymin = min(bv) if ymin is None else min(ymin, min(bv))
    ax.set_ylim((ymin - 0.1) if ymin is not None else 0.0, ymax)
    ax.set_xlim(0, STEP_END)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation CE (nats)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"DUST, dust_target = {target}, population {population}, dust_lr_scale 1: "
                 f"validation CE over the first 500 steps", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sweep_table(runs, population: int, base_value: Optional[float],
                ref_value: Optional[float], md: bool = True, gamma: float = 0.0,
                base_label: str = "hard Top-32 baseline (no DUST steps, same settings)") -> str:
    lines = ["| dust_target \\ sigma | " + " | ".join(f"{s:g}" for s in SIGMAS) + " |",
             "|---|" + "---|" * len(SIGMAS)]
    for target, _ in TARGETS:
        cells = []
        for sigma in SIGMAS:
            run = runs.get((target, sigma, 1, population, md, gamma))
            v = value_at(run["val"], STEP_END) if run else None
            cells.append("--" if run is None else (f"{v:.4f}" if v is not None
                                                   else f"(step {run['last_step']})"))
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    if base_value is not None:
        lines.append(f"| {base_label} | "
                     + " | ".join([f"{base_value:.4f}"] + [""] * (len(SIGMAS) - 1)) + " |")
    if ref_value is not None:
        lines.append("| hard Top-32, the original madrid run (clip 1.0) | "
                     + " | ".join([f"{ref_value:.4f}"] + [""] * (len(SIGMAS) - 1)) + " |")
    return "\n".join(lines)


NOMD_BASE_LABEL = "hard Top-32, code carried, no MD decoupling (baseline: no DUST steps)"


def nomd_panels(runs, population: int, gamma: float, baseline_nomd, out_path: str,
                ymax: float) -> None:
    """Two panels, encoder left and backprop right: the sigma sweep without MD
    decoupling at the given population and gamma against the hard Top-32 run
    without decoupling."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150, sharey=True)
    ramp = matplotlib.colormaps["Oranges"]
    colors = [ramp(0.4 + 0.55 * i / (len(SIGMAS) - 1)) for i in range(len(SIGMAS))]
    ymin = None
    for ax, (target, _) in zip(axes, TARGETS):
        for color, sigma in zip(colors, SIGMAS):
            run = runs.get((target, sigma, 1, population, False, gamma))
            if run is None or not run["val"]:
                continue
            steps = [s for s, _ in run["val"] if s <= STEP_END]
            vals = [v for s, v in run["val"] if s <= STEP_END]
            ax.plot(steps, vals, color=color, lw=2.0, marker="o", ms=3.5,
                    label=f"sigma {sigma:g}")
            ymin = min(vals) if ymin is None else min(ymin, min(vals))
        if baseline_nomd:
            bs = [s for s, _ in baseline_nomd if s <= STEP_END]
            bv = [v for s, v in baseline_nomd if s <= STEP_END]
            ax.plot(bs, bv, color="black", ls="--", lw=1.8, marker="s", ms=3.0,
                    label=NOMD_BASE_LABEL)
            ymin = min(bv) if ymin is None else min(ymin, min(bv))
        ax.set_title(f"dust_target = {target}", fontsize=11)
        ax.set_xlim(0, STEP_END)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("validation CE (nats)")
    for ax in axes:
        ax.set_ylim((ymin - 0.1) if ymin is not None else 0.0, ymax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"DUST without MD decoupling, population {population}, dust_lr_scale 1, "
                 f"gamma {gamma:g}: validation CE over the first 500 steps", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def hard_md_figure(baseline, baseline_nomd, out_path: str, ymax: float) -> None:
    """The hard Top-32 run with and without MD decoupling."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    ymin = None
    for curve, color, ls, label in ((baseline, "black", "--", "hard Top-32, MD decoupling"),
                                    (baseline_nomd, "#b35806", "-", "hard Top-32, no MD decoupling (AdamW, wd 0.1)")):
        if not curve:
            continue
        bs = [s for s, _ in curve if s <= STEP_END]
        bv = [v for s, v in curve if s <= STEP_END]
        ax.plot(bs, bv, color=color, ls=ls, lw=2.0, marker="s", ms=3.0, label=label)
        ymin = min(bv) if ymin is None else min(ymin, min(bv))
    ax.set_ylim((ymin - 0.1) if ymin is not None else 0.0, ymax)
    ax.set_xlim(0, STEP_END)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation CE (nats)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("hard Top-32, code carried: with and without MD decoupling "
                 "(same initialization, no clipping)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def table(runs, target: str, base_value: Optional[float], ref_value: Optional[float]) -> str:
    head = "| sigma \\ dust_lr_scale | " + " | ".join(str(s) for s in SCALES) + " |"
    sep = "|---|" + "---|" * len(SCALES)
    lines = [head, sep]
    best = None
    for sigma in SIGMAS:
        cells = []
        for scale in SCALES:
            run = runs.get((target, sigma, scale, GRID_POPULATION, True, 0.0))
            v = value_at(run["val"], STEP_END) if run else None
            if v is None:
                cells.append("--" if run is None else f"(step {run['last_step']})")
                continue
            cells.append(f"{v:.4f}")
            if best is None or v < best[0]:
                best = (v, sigma, scale)
        lines.append(f"| {sigma:g} | " + " | ".join(cells) + " |")
    if base_value is not None:
        lines.append("| hard Top-32 baseline (no DUST steps, same settings) | " + " | ".join(
            [f"{base_value:.4f}"] + [""] * (len(SCALES) - 1)) + " |")
    if ref_value is not None:
        lines.append("| hard Top-32, the original madrid run (clip 1.0) | " + " | ".join(
            [f"{ref_value:.4f}"] + [""] * (len(SCALES) - 1)) + " |")
    out = "\n".join(lines)
    if best is not None:
        out += f"\n\nbest cell: sigma {best[1]:g}, dust_lr_scale {best[2]}: {best[0]:.4f}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", required=True)
    ap.add_argument("--baseline", default=None,
                    help="metrics.jsonl of the grid's hard Top-32 baseline run")
    ap.add_argument("--baseline-nomd", default=None,
                    help="metrics.jsonl of the hard Top-32 run without MD decoupling")
    ap.add_argument("--ref", default=None,
                    help="metrics.jsonl of the original madrid hard Top-32 run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ymax", type=float, default=6.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs_dir)
    baseline = val_curve(read_jsonl(args.baseline)) if args.baseline else []
    base_value = value_at(baseline, STEP_END) if baseline else None
    baseline_nomd = val_curve(read_jsonl(args.baseline_nomd)) if args.baseline_nomd else []
    base_nomd_value = value_at(baseline_nomd, STEP_END) if baseline_nomd else None
    ref_value = value_at(val_curve(read_jsonl(args.ref)), STEP_END) if args.ref else None
    for target, tag in TARGETS:
        present = {k: v for k, v in runs.items()
                   if k[0] == target and k[3] == GRID_POPULATION and k[4] and k[5] == 0.0}
        if not present:
            print(f"[{target}] no runs found")
            continue
        path = os.path.join(args.out, f"hp500_val_{target}.png")
        figure(runs, target, baseline, path, args.ymax)
        n_done = sum(1 for r in present.values() if r["finished"])
        print(f"\n## {target}: {len(present)} runs ({n_done} finished) -> {path}\n")
        print(table(runs, target, base_value, ref_value))
        print("\nruns:")
        for key in sorted(present):
            r = present[key]
            flag = "" if r["finished"] else f"  [unfinished, last step {r['last_step']}]"
            print(f"  sigma {key[1]:g} lr {key[2]:>3}: {r['name']}{flag}")
    populations = sorted({k[3] for k in runs if k[3] != GRID_POPULATION})
    for population in populations:
        md_keys = [k for k in runs if k[3] == population and k[4] and k[5] == 0.0]
        if md_keys:
            for target, tag in TARGETS:
                if not any(k[0] == target for k in md_keys):
                    continue
                path = os.path.join(args.out, f"hp500_p{population}_val_{target}.png")
                sweep_figure(runs, target, population, baseline, path, args.ymax)
                print(f"\n## population {population}, MD, {target} -> {path}")
            print(f"\n## population {population}, MD decoupling, dust_lr_scale 1: "
                  "validation CE at step 500\n")
            print(sweep_table(runs, population, base_value, ref_value))
            print("\nruns:")
            for key in sorted(md_keys):
                r = runs[key]
                flag = "" if r["finished"] else f"  [unfinished, last step {r['last_step']}]"
                print(f"  {key[0]} sigma {key[1]:g}: {r['name']}{flag}")
        gammas = sorted({k[5] for k in runs if k[3] == population and not k[4]})
        for gamma in gammas:
            gtag = f"{gamma:g}".replace(".", "p")
            path = os.path.join(args.out, f"hp500_p{population}_nomd_g{gtag}_val.png")
            nomd_panels(runs, population, gamma, baseline_nomd, path, args.ymax)
            print(f"\n## population {population}, no MD decoupling, gamma {gamma:g} -> {path}")
            print(f"\n## population {population}, no MD decoupling, gamma {gamma:g}, "
                  "dust_lr_scale 1: validation CE at step 500\n")
            print(sweep_table(runs, population, base_nomd_value, None, md=False, gamma=gamma,
                              base_label="hard Top-32, no MD decoupling (no DUST steps, same settings)"))
            print("\nruns:")
            for key in sorted(k for k in runs if k[3] == population and not k[4] and k[5] == gamma):
                r = runs[key]
                flag = "" if r["finished"] else f"  [unfinished, last step {r['last_step']}]"
                print(f"  {key[0]} sigma {key[1]:g}: {r['name']}{flag}")
    if baseline and baseline_nomd:
        path = os.path.join(args.out, "hp500_hard_md_vs_nomd.png")
        hard_md_figure(baseline, baseline_nomd, path, args.ymax)
        print(f"\n## hard Top-32 with vs without MD decoupling -> {path}")
        print(f"\n| run | validation CE at step 500 |\n|---|---|\n"
              f"| hard Top-32, MD decoupling | {base_value:.4f} |\n"
              f"| hard Top-32, no MD decoupling (AdamW, wd 0.1) | {base_nomd_value:.4f} |"
              if base_value is not None and base_nomd_value is not None else "")


if __name__ == "__main__":
    main()
