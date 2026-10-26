"""Write the run configs and the GPU plans of the small-model DUST study.

    python scripts/make_small_dust_configs.py [--steps 10000]

The grid (2026-10-26): for K in (4, 8, 16), in that order, the hard Top-K
run and DUST with 128, 512 and 2048 draws per bottleneck
(dust_population = 4 x draws, 4 bottlenecks), J = N_features - K, all from
configs/small_dust/base*.yaml.  Every run trains --steps steps (constant LR,
so max_steps only sets the length), logs every 2 steps (one hard and one
DUST step per window under [1, 1]), validates every 20 and checkpoints every
500, all kept (the backup watcher uploads every one to Drive; a hard step
takes 67 ms, so pruning would outrun the watcher).

Two boxes, one plan each (read by scripts/gpu_phases.sh), K = 4 only
since 2026-10-26 14:00 UTC (the K = 8 and 16 configs are written but not
planned):

* warsaw (8 GPUs), plan_warsaw.txt, runs ``wa_*``: the hard run on GPU 0
  alongside 128 draws split over GPUs 1-7, then 512 draws over all 8, then
  the schedule sweep at 512 draws over all 8, one run after another:
  dust_schedule [10, 1], [30, 1], [90, 1] (``wa_k4_dust_p512_h10`` etc.:
  10, 30, 90 hard steps per DUST step), then [30, 5] and [90, 10]
  (``wa_k4_dust_p512_h30_d5``, ``_h90_d10``).  The last two share one phase
  (a chain on the same GPUs): they were added while gpu_phases.sh was
  running, and it fixes its list of phase numbers at start.
* lyon (16 GPUs), plan_lyon.txt, runs ``ly_*``: 2048 draws over all 16
  (stopped at 4000 steps by request).

DUST step times on 5090s (2026-10-26): about 0.17 s + 38.4 ms x (draws per
GPU); 128 over 7 GPUs ~0.9 s, 512 over 8 ~2.6 s, 2048 over 16 ~5 s.
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "configs", "small_dust")
N_FEATURES = 512
N_BOTTLENECKS = 4
K_ORDER = (4, 8, 16)
PLAN_K = (4,)
#: the schedule sweep at 512 draws (K = 4): dust_schedule [n_hard, n_dust]
#: per run, grouped by phase (runs of one phase run one after another)
SWEEP_PHASES = (((10, 1),), ((30, 1),), ((90, 1),), ((30, 5), (90, 10)))


def name(prefix: str, k: int, draws: int = 0) -> str:
    return f"{prefix}_k{k}_hard" if not draws else f"{prefix}_k{k}_dust_p{draws}"


def write(prefix: str, k: int, draws: int, steps: int, n_hard: int = 1,
          n_dust: int = 1) -> str:
    sched = (n_hard, n_dust) != (1, 1)
    run = name(prefix, k, draws)
    if sched:
        run += f"_h{n_hard}" + (f"_d{n_dust}" if n_dust != 1 else "")
    base = "base_dust.yaml" if draws else "base.yaml"
    what = (f"DUST, {draws} draws per bottleneck (dust_population {N_BOTTLENECKS * draws})"
            if draws else "hard Top-K baseline")
    if sched:
        what += (f", dust_schedule [{n_hard}, {n_dust}] ({n_hard} hard steps, then "
                 f"{n_dust} DUST step{'s' if n_dust > 1 else ''})")
    lines = [
        f"# Small-model DUST study (configs/small_dust/{base}): K = {k}, {what}.",
        "# Written by scripts/make_small_dust_configs.py.",
        "_base_:",
        f"  - {base}",
        "",
        "train:",
        f"  run_name: {run}",
        f"  max_steps: {steps}",
        "  log_every_steps: 2",
        "  validate_every_steps: 20",
        "  checkpoint_every_steps: 500",
        f"  keep_last_checkpoints: {max(1, steps // 500)}",
        "",
        "activation_bottleneck:",
        f"  k: {k}",
        f"  j: {N_FEATURES - k}",
    ]
    if draws:
        lines.append(f"  dust_population: {N_BOTTLENECKS * draws}")
    if sched:
        lines.append(f"  dust_schedule: [{n_hard}, {n_dust}]")
    rel = f"configs/small_dust/{run}.yaml"
    with open(os.path.join(OUT, f"{run}.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rel


def gpus(ids) -> str:
    return ",".join(str(i) for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    args = ap.parse_args()
    header = ["# phase  gpus  config   (scripts/gpu_phases.sh: phases in order; within a",
              "# phase one chain per GPU set, jobs on the same set one after another,",
              "# sets in parallel)"]
    warsaw, lyon = list(header), list(header)
    phase_w = phase_l = 0
    for k in K_ORDER:
        planned = k in PLAN_K
        cfgs = [write('wa', k, 0, args.steps), write('wa', k, 128, args.steps),
                write('wa', k, 512, args.steps), write('ly', k, 2048, args.steps)]
        if not planned:
            continue
        warsaw += [f"{phase_w + 1} 0 {cfgs[0]}", f"{phase_w + 1} {gpus(range(1, 8))} {cfgs[1]}",
                   f"{phase_w + 2} {gpus(range(8))} {cfgs[2]}"]
        phase_w += 2
        lyon += [f"{phase_l + 1} {gpus(range(16))} {cfgs[3]}"]
        phase_l += 1
    for group in SWEEP_PHASES:
        phase_w += 1
        warsaw += [f"{phase_w} {gpus(range(8))} {write('wa', 4, 512, args.steps, h, d)}"
                   for h, d in group]
    for box, plan in (("warsaw", warsaw), ("lyon", lyon)):
        with open(os.path.join(OUT, f"plan_{box}.txt"), "w") as f:
            f.write("\n".join(plan) + "\n")
    n_sweep = sum(len(g) for g in SWEEP_PHASES)
    print(f"wrote {4 * len(K_ORDER) + n_sweep} configs, plan_warsaw.txt and "
          f"plan_lyon.txt for {args.steps} steps")


if __name__ == "__main__":
    main()
