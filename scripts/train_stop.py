"""Train a config but stop cleanly after --stop-step optimizer steps.

``train.max_steps`` (and with it every schedule: lr cosine, temperature
anneal) is deliberately left untouched, so the trajectory is exactly the
trajectory of a full-length run -- stopping is an ``on_step`` exception,
the same trick ``analysis/probe_early_training.py`` uses.  For short
falsification runs that must be comparable to full 20k-step runs.

Usage:
    python scripts/train_stop.py --config CFG --stop-step 6000 [key=val ...]
"""

from __future__ import annotations

import argparse

from wsparse.config import load_config
from wsparse.train import train


class StopRun(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stop-step", type=int, required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    def stop(step, model, bottleneck, optimizer):
        if step >= args.stop_step:
            raise StopRun

    try:
        train(cfg, on_step=stop)
    except StopRun:
        print(f"[train_stop] stopped at step {args.stop_step} as planned")


if __name__ == "__main__":
    main()
