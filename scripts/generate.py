"""Sample from a checkpoint, optionally with hard-pruned weights.

    python scripts/generate.py --ckpt runs/ltp_small/latest.pt \
        --prompt "Once upon a time" --hard
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wsparse.tokenizer import build_tokenizer  # noqa: E402
from wsparse.train import load_for_inference  # noqa: E402
from wsparse.utils import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--hard", action="store_true", help="use binary masks (the actually pruned network)"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, controller = load_for_inference(args.ckpt, device=str(device))
    tokenizer = build_tokenizer(cfg.data)
    if controller.enabled:
        stats = controller.stats()
        print(
            f"[gen] {cfg.sparsity.method} sparsity: beta={stats['sparsity/beta']:.3g} "
            f"density_hard={stats['sparsity/density_hard']:.4f} "
            f"density_soft={stats['sparsity/density_soft']:.4f}"
        )

    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    ctx = controller.hard_mask() if (args.hard and controller.enabled) else _null()
    with ctx:
        for i in range(args.samples):
            out = model.generate(
                ids, args.tokens, temperature=args.temperature, top_k=args.top_k or None
            )
            print(f"\n--- sample {i + 1} ---\n{tokenizer.decode(out[0].tolist())}")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
