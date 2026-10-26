"""Write the run configs and the GPU plan of the small-model DUST study.

    python scripts/make_small_dust_configs.py --steps N [--prefix wa]

For K in (16, 8, 4): the hard Top-K run and the DUST runs with 128, 512,
2048 and 8192 draws per bottleneck (dust_population = 4 x draws, 4
bottlenecks), J = N_features - K, all from configs/small_dust/base*.yaml.
Every run trains N steps (constant LR, so max_steps only sets the length),
logs every 2 steps (one hard and one DUST step per window under [1, 1]),
validates every 10 and checkpoints at N/2 and N (the last two kept).

The plan (configs/small_dust/plan.txt, read by scripts/gpu_phases.sh) runs
in phases on 8 GPUs; 2048 draws are split over 4 GPUs and 8192 over 8
(DustDrawGroup), the rest take one GPU each.  Step times on a 5090
(2026-10-26): hard 67 ms, DUST 5.0 s (128), 19.5 s (512), 19.8 s (2048 / 4
GPUs), 39.8 s (8192 / 8 GPUs); the plan takes about 89 N seconds.
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "configs", "small_dust")
N_FEATURES = 512
N_BOTTLENECKS = 4
K_LIST = (16, 8, 4)
DRAWS = (128, 512, 2048, 8192)
GPUS = {128: "1", 512: "1", 2048: "4", 8192: "8"}


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
        "  validate_every_steps: 10",
        f"  checkpoint_every_steps: {max(1, steps // 2)}",
        "  keep_last_checkpoints: 2",
        "",
        "activation_bottleneck:",
        f"  k: {k}",
        f"  j: {N_FEATURES - k}",
    ]
    if draws:
        lines.append(f"  dust_population: {N_BOTTLENECKS * draws}")
    path = os.path.join(OUT, f"{run}.yaml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--prefix", default="wa")
    args = ap.parse_args()
    for k in K_LIST:
        write(args.prefix, k, 0, args.steps)
        for d in DRAWS:
            write(args.prefix, k, d, args.steps)
    c = lambda k, d=0: f"configs/small_dust/{name(args.prefix, k, d)}.yaml"  # noqa: E731
    all8, four = "0,1,2,3,4,5,6,7", "0,1,2,3"
    plan = [
        "# phase  gpus  config   (gpu_phases.sh: phases in order; within a phase one",
        "# chain per GPU set, jobs on the same set one after another, sets in parallel)",
        f"1 {four} {c(16, 2048)}", f"1 4 {c(16, 512)}", f"1 5 {c(16, 128)}",
        f"1 6 {c(16)}", f"1 7 {c(8)}", f"1 7 {c(8, 128)}",
        f"2 {all8} {c(16, 8192)}",
        f"3 {four} {c(8, 2048)}", f"3 4 {c(8, 512)}", f"3 5 {c(4)}", f"3 5 {c(4, 128)}",
        f"3 6 {c(4, 512)}",
        f"4 {all8} {c(8, 8192)}",
        f"5 {four} {c(4, 2048)}",
        f"6 {all8} {c(4, 8192)}",
    ]
    with open(os.path.join(OUT, "plan.txt"), "w") as f:
        f.write("\n".join(plan) + "\n")
    print(f"wrote {len(K_LIST) * (1 + len(DRAWS))} configs and plan.txt for {args.steps} steps")


if __name__ == "__main__":
    main()
