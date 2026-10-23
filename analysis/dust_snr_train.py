"""Train an SNR-probe model for a few steps with a checkpoint at init.

    python analysis/dust_snr_train.py --config configs/cr_dust/snrprobe/<run>.yaml [--steps 20] [overrides]

The trainer's own checkpointing (train.checkpoint_every_steps, here 2, all
kept) writes ckpt_step2.pt ... ckpt_step<steps>.pt, i.e. the model after every
DUST step of the [1, 1] schedule; this driver adds ckpt_step0.pt, the model at
initialization, through the ``on_step`` hook (called before the first update),
and stops the run cleanly through ``should_stop`` once ``--steps`` optimizer
steps are done, with train.max_steps and every schedule untouched.
"""

from __future__ import annotations

import argparse
import os

from wsparse.config import load_config
from wsparse.train import save_checkpoint, train, unwrap_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("overrides", nargs="*")
    args, unknown = ap.parse_known_args()
    cfg = load_config(args.config, list(unknown) + args.overrides)
    run_dir = os.path.join(cfg.train.out_dir, cfg.train.run_name)
    state = {"step": -1}

    def on_step(step, model, bottleneck, optimizer):
        state["step"] = step
        if step == 0:
            path = os.path.join(run_dir, "ckpt_step0.pt")
            save_checkpoint(path, cfg, unwrap_model(model), optimizer, 0)
            print(f"[dust_snr_train] saved {path} (initialization)")

    def should_stop():
        # polled after each step's checkpoint: the step index of the last
        # completed step is state["step"]
        return (f"probe run complete after {args.steps} steps"
                if state["step"] >= args.steps - 1 else None)

    summary = train(cfg, on_step=on_step, should_stop=should_stop)
    print(f"[dust_snr_train] done: stopped_at {summary.get('stopped_at')}; checkpoints: "
          + " ".join(sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_step"))))


if __name__ == "__main__":
    main()
