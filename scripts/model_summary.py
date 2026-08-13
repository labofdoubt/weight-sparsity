"""Print a parameter breakdown for a config (no data or GPU needed).

    python scripts/model_summary.py --config configs/ltp_150m.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wsparse.config import load_config  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.sparsity import apply_sparsity  # noqa: E402
from wsparse.utils import human  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    model = build_model(cfg.model)
    controller = apply_sparsity(model, cfg.sparsity, max_steps=cfg.train.max_steps)

    total = model.num_parameters()
    non_emb = model.num_parameters(non_embedding=True)
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()

    print(f"config            : {args.config}")
    print(f"layers/d_model    : {cfg.model.n_layers} / {cfg.model.d_model} "
          f"({cfg.model.n_heads} heads, d_mlp={cfg.model.d_mlp})")
    print(f"vocab / seq_len   : {cfg.model.vocab_size} / {cfg.model.max_seq_len}")
    print(f"total parameters  : {total:,} ({human(total)})")
    print(f"  embeddings      : {emb:,} ({human(emb)})")
    print(f"  transformer     : {non_emb:,} ({human(non_emb)})")
    if controller.enabled:
        maskable = controller.total_maskable
        print(f"maskable weights  : {maskable:,} ({human(maskable)}) "
              f"= {maskable / total:.1%} of all parameters, "
              f"{maskable / non_emb:.1%} of the transformer")
        extra = sum(p.numel() for p in controller.mask_parameters())
        print(f"sparsity params   : {extra:,} ({cfg.sparsity.method}; "
              f"training-time only, not part of the model)")
        if cfg.sparsity.target_density is not None:
            kept = sum(
                controller.target_density[n] * layer.mask_numel for n, layer in controller.layers
            )
            print(f"target kept       : {kept:,.0f} of {maskable:,} maskable "
                  f"({kept / maskable:.1%}); model would be {total - maskable + kept:,.0f} params")
    else:
        print("sparsity          : disabled")


if __name__ == "__main__":
    main()
