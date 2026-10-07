"""Train a config but abort cleanly once the run has clearly diverged.

For campaign runs where some configurations are expected to destabilize:
letting a dead run finish its schedule wastes a GPU for hours.  Divergence is
judged from the training CE the run itself logs, so nothing in the training
loop changes -- the wrapper observes ``Logger.log``.

Rules (all thresholds are flags):

* a non-finite ``train/ce`` or ``val/ce`` aborts immediately (and, for
  laplace_policy runs, a non-finite ``train_stochastic/ce``,
  ``train_deterministic_probe/ce``, ``val_stochastic/ce`` or
  ``val_deterministic/ce``);
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


class StoppedEarly(Exception):
    def __init__(self, step: int):
        super().__init__(f"reached --stop-step at {step}")
        self.step = step


class Diverged(Exception):
    """Retained as a record type; the guard no longer raises.

    Stopping by exception cannot work under DDP -- the raising rank leaves the
    others hanging in the next gradient all-reduce -- so the guard sets a flag
    that train()'s ``should_stop`` hook polls and all-reduces, and every rank
    breaks in the same step.  The single-process path uses the same mechanism.
    """

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
    # Stop the RUN at this step while leaving cfg.train.max_steps -- and with
    # it every schedule (LR cosine, temperature) -- untouched, so the
    # trajectory up to the stop is identical to a full-length run's and
    # step-matched comparisons against full-length references stay valid.
    # 0 disables.  Contrast with overriding train.max_steps, which reshapes
    # the schedules and produces a run comparable to nothing existing.
    ap.add_argument("--stop-step", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    # dotted --section.field=value overrides arrive as unknown flags; collect
    # them like wsparse.train's own main() does (apply_overrides rejects any
    # stray token without '=', so a mistyped guard flag still fails loudly)
    args, unknown = ap.parse_known_args()
    cfg = load_config(args.config, list(unknown) + args.overrides)
    run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    is_main = int(os.environ.get("RANK", "0")) == 0

    state = {"best": math.inf, "last": None, "over_ceiling_since": None,
             "over_margin_since": None,
             # set instead of raising: (kind, step, reason)
             "stop": None}

    orig_log = utils.Logger.log

    def guarded_log(self, step, metrics, console=""):
        orig_log(self, step, metrics, console)
        # laplace_policy logs its sampled / deterministic CEs under the extra
        # keys; a non-finite value in any actual CE aborts (the support-term
        # diagnostics are not CEs and are deliberately not watched)
        for key in ("train/ce", "val/ce", "train_stochastic/ce",
                    "train_deterministic_probe/ce", "val_stochastic/ce",
                    "val_deterministic/ce"):
            v = metrics.get(key)
            if v is not None and not math.isfinite(v):
                state["stop"] = ("diverged", step, f"{key} is non-finite ({v})")
                return
        if state["stop"] is not None:
            return  # already stopping; don't overwrite the first reason
        if args.stop_step and step >= args.stop_step:
            state["stop"] = ("stopped", step, f"reached --stop-step {args.stop_step}")
            return
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
                state["stop"] = ("diverged", step,
                                 f"train/ce > {args.ceiling} for "
                                 f"{step - state['over_ceiling_since']} steps")
        else:
            state["over_ceiling_since"] = None

        if ce > state["best"] + args.margin:
            if state["over_margin_since"] is None:
                state["over_margin_since"] = step
            elif step - state["over_margin_since"] >= args.margin_steps:
                state["stop"] = ("diverged", step,
                                 f"train/ce > best+{args.margin} "
                                 f"(best {state['best']:.3f}) for "
                                 f"{step - state['over_margin_since']} steps")
        else:
            state["over_margin_since"] = None

    utils.Logger.log = guarded_log

    def should_stop():
        # polled by train() once per optimizer step on every rank; only rank 0
        # has a real Logger (and so a live guard) -- the other ranks return
        # None and learn about the stop from the all-reduce inside train()
        return state["stop"][2] if state["stop"] is not None else None

    try:
        train(cfg, should_stop=should_stop)
    finally:
        utils.Logger.log = orig_log

    if state["stop"] is not None and is_main:
        kind, step, reason = state["stop"]
        if kind == "stopped":
            print(f"[train_guard] STOPPED {cfg.train.run_name} at step {step} "
                  f"(--stop-step {args.stop_step}; best {state['best']:.4f})")
            payload = {"step": step, "stop_step": args.stop_step,
                       "best_train_ce": state["best"],
                       "last_train_ce": state["last"]}
        else:
            print(f"[train_guard] DIVERGED {cfg.train.run_name} at step {step}: "
                  f"{reason} (best {state['best']:.4f}, last {state['last']})")
            payload = {"step": step, "reason": reason,
                       "best_train_ce": state["best"],
                       "last_train_ce": state["last"]}
        try:
            with open(os.path.join(run_dir, f"{kind}.json"), "w") as f:
                json.dump(payload, f, indent=1)
        except OSError as e:
            print(f"[train_guard] could not write {kind}.json: {e}")


if __name__ == "__main__":
    main()
