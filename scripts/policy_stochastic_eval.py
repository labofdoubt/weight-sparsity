"""Many-draw stochastic validation of laplace_policy checkpoints.

Training validates with ``train.policy_val_samples`` draws (two in the
policy_cr campaign) to keep the overhead small; this re-evaluates finished
checkpoints with more draws on the SAME validation windows: the clean
deterministic Top-K CE (``val/ce``) and the mean over ``--samples`` sampled
draws at the width the schedule had at the checkpoint (``tau(step - 1)``, the
value the training-time validation of that step used), with the draws' sample
standard deviation.  The draws use the evaluation generator re-seeded from
``--seed`` exactly as ``wsparse.train.evaluate_stochastic`` does, so draws
1-2 of a run reproduce its own training-time ``val_stochastic`` numbers up to
bf16 non-determinism.

    python scripts/policy_stochastic_eval.py --ckpt RUN/ckpt_step20000.pt \
        [--samples 8] [--batches 40] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.data import build_streams, load_meta  # noqa: E402
from wsparse.train import evaluate, evaluate_stochastic, load_for_inference  # noqa: E402
from wsparse.utils import resolve_dtype  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batches", type=int, default=0, help="0: the run's train.val_batches")
    ap.add_argument("--seed", type=int, default=-1, help="-1: the run's train.policy_val_seed")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    results = {}
    for path in args.ckpt:
        model, cfg, bottleneck = load_for_inference(path, device=str(device))
        step = int(torch.load(path, map_location="cpu", weights_only=False)["step"])
        tau = bottleneck.set_step(max(0, step - 1))
        if args.data_dir:
            cfg.data.data_dir = args.data_dir
        meta = load_meta(cfg.data.data_dir)
        cfg.model.vocab_size = int(meta["vocab_size"])
        _, val_stream = build_streams(cfg.data, seed=cfg.train.seed)
        dtype = resolve_dtype(cfg.train.dtype, device)
        micro = cfg.train.micro_batch_size
        batches = args.batches or cfg.train.val_batches
        seed = cfg.train.policy_val_seed if args.seed < 0 else args.seed
        clean = evaluate(model, val_stream, micro, batches, device, dtype)
        # one call, the draws consecutive from one generator seeded once: exactly
        # what the training-time validation does, so its draws are the first two
        sto = evaluate_stochastic(model, val_stream, micro, batches, device, dtype,
                                  args.samples, seed)
        std = sto.get("mc_std", float("nan"))
        n = int(sto["samples"])
        sto, draws = sto["ce"], n
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        results[name] = {"ckpt": path, "step": step, "tau": tau, "batches": batches,
                         "samples": args.samples, "seed": seed, "clean_ce": clean["ce"],
                         "stochastic_ce": sto, "stochastic_std": std,
                         "stochastic_sem": std / n ** 0.5 if n > 1 else None,
                         "gap": sto - clean["ce"]}
        print(f"{name:36s} step {step:6d} tau {tau:.4g} clean {clean['ce']:.4f} "
              f"stochastic {sto:.4f} (+-{std:.4f} over {n} draws) gap {sto - clean['ce']:+.4f}")
        del model
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
