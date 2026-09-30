"""Print a parameter breakdown for a config (no data or GPU needed).

    python scripts/model_summary.py --config configs/bn_hard.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from wsparse.config import load_config  # noqa: E402
from wsparse.model import build_model  # noqa: E402
from wsparse.bottleneck import apply_activation_bottleneck  # noqa: E402
from wsparse.utils import human  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    model = build_model(cfg.model)
    bottleneck = apply_activation_bottleneck(model, cfg.activation_bottleneck)

    total = model.num_parameters()
    non_emb = model.num_parameters(non_embedding=True)
    emb = model.tok_emb.weight.numel()
    if model.pos_emb is not None:
        emb += model.pos_emb.weight.numel()

    print(f"config            : {args.config}")
    print(f"layers/d_model    : {cfg.model.n_layers} / {cfg.model.d_model} "
          f"({cfg.model.n_heads} heads, d_mlp={cfg.model.d_mlp})")
    print(f"vocab / seq_len   : {cfg.model.vocab_size} / {cfg.model.max_seq_len}")
    print(f"total parameters  : {total:,} ({human(total)})")
    print(f"  embeddings      : {emb:,} ({human(emb)})")
    print(f"  transformer     : {non_emb:,} ({human(non_emb)})")

    if bottleneck.enabled:
        cb = cfg.activation_bottleneck
        added = bottleneck.n_parameters
        dense = total - added
        n_sets = len(bottleneck.projection_owners())  # 1 under share_projections
        print(f"bottleneck        : {len(bottleneck.layers)} layers, {n_sets} x "
              f"(2 x {cfg.model.d_model} x {cb.n_features})"
              f"{' shared by all' if cb.share_projections else ''} "
              f"= {added:,} ({human(added)}) "
              f"added, {added / dense:+.1%} over the {dense:,}-param dense model")
        print(f"  active / layer  : K={cb.k} of N={cb.n_features} "
              f"({cb.k / cb.n_features:.1%} of features, {cb.k / cfg.model.d_model:.2f}x d_model)")
        print(f"  gradient pool   : K+J={cb.k + cb.j} "
              f"({(cb.k + cb.j) / cb.n_features:.1%}), T={cb.temperature:g}, "
              f"surrogate={cb.surrogate_mode}")
    else:
        print("bottleneck        : disabled")


if __name__ == "__main__":
    main()
