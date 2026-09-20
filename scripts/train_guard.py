"""Train a config but abort cleanly once the run has clearly diverged.

For campaign runs where some configurations are expected to destabilize:
letting a dead run finish its schedule wastes a GPU for hours.  Divergence is
judged from the training CE the run itself logs, so nothing in the training
loop changes -- the wrapper observes ``Logger.log``.

Rules (all thresholds are flags):

* a non-finite ``train/ce`` or ``val/ce`` aborts immediately;
* ``train/ce`` above ``--ceiling`` (default 7.0) continuously for
  ``--ceiling-steps`` (default 300) aborts, but only after ``--min-step``
  (default 1500) so the early descent from ln(vocab) is exempt;
* ``train/ce`` above best-so-far + ``--margin`` (default 2.5) continuously
  for ``--margin-steps`` (default 800) aborts.  The window is long enough
  that a burst-and-recover excursion of a few hundred steps does not trip it.

On abort the wrapper writes ``diverged.json`` into the run directory (step,
reason, best and last CE) and exits 0, so a job queue treats the run as
finished and moves on.

Usage:
    python scripts/train_guard.py --config CFG [key=val ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os

from wsparse.config import load_config
from wsparse.train import train
from wsparse import utils


class Diverged(Exception):
    def __init__(self, step: int, reason: str):
        super().__init__(reason)
        self.step = step
        self.reason = reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ceiling", type=float, default=7.0)
    ap.add_argument("--ceiling-steps", type=int, default=300)
    ap.add_argument("--margin", type=float, default=2.5)
    ap.add_argument("--margin-steps", type=int, default=800)
    ap.add_argument("--min-step", type=int, default=1500)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)

    state = {"best": math.inf, "last": None, "over_ceiling_since": None,
             "over_margin_since": None}

    orig_log = utils.Logger.log

    def guarded_log(self, step, metrics, console=""):
        orig_log(self, step, metrics, console)
        for key in ("train/ce", "val/ce"):
            v = metrics.get(key)
            if v is not None and not math.isfinite(v):
                raise Diverged(step, f"{key} is non-finite ({v})")
        ce = metrics.get("train/ce")
        if ce is None:
            return
        state["last"] = ce
        if ce < state["best"]:
            state["best"] = ce

        if ce > args.ceiling and step >= args.min_step:
            if state["over_ceiling_since"] is None:
                state["over_ceiling_since"] = step
            elif step - state["over_ceiling_since"] >= args.ceiling_steps:
                raise Diverged(step, f"train/ce > {args.ceiling} for "
                                     f"{step - state['over_ceiling_since']} steps")
        else:
            state["over_ceiling_since"] = None

        if ce > state["best"] + args.margin:
            if state["over_margin_since"] is None:
                state["over_margin_since"] = step
            elif step - state["over_margin_since"] >= args.margin_steps:
                raise Diverged(step, f"train/ce > best+{args.margin} "
                                     f"(best {state['best']:.3f}) for "
                                     f"{step - state['over_margin_since']} steps")
        else:
            state["over_margin_since"] = None

    utils.Logger.log = guarded_log
    try:
        train(cfg)
    except Diverged as d:
        print(f"[train_guard] DIVERGED {cfg.train.run_name} at step {d.step}: "
              f"{d.reason} (best {state['best']:.4f}, last {state['last']})")
        try:
            with open(os.path.join(run_dir, "diverged.json"), "w") as f:
                json.dump({"step": d.step, "reason": d.reason,
                           "best_train_ce": state["best"],
                           "last_train_ce": state["last"]}, f, indent=1)
        except OSError as e:
            print(f"[train_guard] could not write diverged.json: {e}")
    finally:
        utils.Logger.log = orig_log


if __name__ == "__main__":
    main()
