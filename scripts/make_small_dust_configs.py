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

Two boxes, one plan each (read by scripts/gpu_phases.sh):

* warsaw (8 GPUs), plan_warsaw.txt, runs ``wa_*``: per K, the hard run on
  GPU 0 alongside 128 draws split over GPUs 1-7, then 512 draws over all 8.
* lyon (16 GPUs), plan_lyon.txt, runs ``ly_*``: per K, 2048 draws over all 16.

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


def name(prefix: str, k: int, draws: int = 0) -> str:
    return f"{prefix}_k{k}_hard" if not draws else f"{prefix}_k{k}_dust_p{draws}"


def write(prefix: str, k: int, draws: int, steps: int) -> str:
    run = name(prefix, k, draws)
    base = "base_dust.yaml" if draws else "base.yaml"
    what = (f"DUST, {draws} draws per bottleneck (dust_population {N_BOTTLENECKS * draws})"
            if draws else "hard Top-K baseline")
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
    for i, k in enumerate(K_ORDER):
        warsaw += [f"{2 * i + 1} 0 {write('wa', k, 0, args.steps)}",
                   f"{2 * i + 1} {gpus(range(1, 8))} {write('wa', k, 128, args.steps)}",
                   f"{2 * i + 2} {gpus(range(8))} {write('wa', k, 512, args.steps)}"]
        lyon += [f"{i + 1} {gpus(range(16))} {write('ly', k, 2048, args.steps)}"]
    for box, plan in (("warsaw", warsaw), ("lyon", lyon)):
        with open(os.path.join(OUT, f"plan_{box}.txt"), "w") as f:
            f.write("\n".join(plan) + "\n")
    print(f"wrote {4 * len(K_ORDER)} configs, plan_warsaw.txt and plan_lyon.txt "
          f"for {args.steps} steps")


if __name__ == "__main__":
    main()
