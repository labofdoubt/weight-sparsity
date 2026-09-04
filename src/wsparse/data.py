"""TinyStories data preparation and batching.

``python -m wsparse.data --config configs/base.yaml`` downloads TinyStories,
tokenizes it and writes flat ``uint16`` token streams::

    data/tinystories/train.bin
    data/tinystories/val.bin
    data/tinystories/meta.json

Training then samples random windows out of those memmaps (nanoGPT style).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import Config, DataConfig, load_config
from .tokenizer import build_tokenizer, train_bpe_tokenizer

META_NAME = "meta.json"


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #


def prepare(cfg: DataConfig, force: bool = False) -> Dict:
    """Download + tokenize TinyStories into ``cfg.data_dir``."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    meta_path = os.path.join(cfg.data_dir, META_NAME)
    if os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[data] reusing existing dataset in {cfg.data_dir}: {meta}")
        return meta

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("data preparation needs `datasets` (pip install datasets)") from exc

    print(f"[data] loading {cfg.dataset} ...")
    ds = load_dataset(cfg.dataset)
    train_ds = ds["train"]
    if "validation" in ds and cfg.val_fraction == 0.0:
        val_ds = ds["validation"]
    else:
        frac = cfg.val_fraction or 0.005
        split = train_ds.train_test_split(test_size=frac, seed=0)
        train_ds, val_ds = split["train"], split["test"]

    if cfg.tokenizer == "bpe" and not os.path.exists(
        os.path.join(cfg.tokenizer_path, "tokenizer.json")
    ):
        print(f"[data] training a {cfg.bpe_vocab_size}-token BPE on TinyStories ...")
        n_docs = min(cfg.bpe_train_docs, len(train_ds))
        # .select + streaming iteration; indexing the whole column would pull
        # every story into memory at once
        subset = train_ds.select(range(n_docs))
        train_bpe_tokenizer(
            (row["text"] for row in subset),
            cfg.tokenizer_path,
            vocab_size=cfg.bpe_vocab_size,
            max_docs=n_docs,
        )
    tokenizer = build_tokenizer(cfg)
    if tokenizer.vocab_size >= 2**16:
        raise ValueError(
            f"vocab_size={tokenizer.vocab_size} does not fit in uint16; use a smaller tokenizer"
        )

    def tokenize(batch):
        ids = [tokenizer.encode(t, add_eos=True) for t in batch["text"]]
        return {"ids": ids, "len": [len(i) for i in ids]}

    counts = {}
    for name, split_ds in (("train", train_ds), ("val", val_ds)):
        tokenized = split_ds.map(
            tokenize,
            batched=True,
            batch_size=1000,
            remove_columns=split_ds.column_names,
            num_proc=max(1, cfg.num_proc),
            desc=f"tokenizing {name}",
        )
        total = int(np.sum(tokenized["len"], dtype=np.int64))
        path = os.path.join(cfg.data_dir, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        offset = 0
        shards = max(1, min(64, len(tokenized) // 10000))
        for shard in range(shards):
            batch = tokenized.shard(num_shards=shards, index=shard, contiguous=True)
            flat = np.concatenate([np.asarray(x, dtype=np.uint16) for x in batch["ids"]])
            arr[offset : offset + len(flat)] = flat
            offset += len(flat)
        arr.flush()
        del arr
        counts[name] = total
        print(f"[data] wrote {path}: {total:,} tokens")

    meta = {
        "dataset": cfg.dataset,
        "tokenizer": cfg.tokenizer,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_id": int(tokenizer.eos_id),
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(data_dir: str) -> Dict:
    path = os.path.join(data_dir, META_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m wsparse.data --config <cfg>` first"
        )
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #


class TokenStream:
    """Samples random ``seq_len + 1`` windows from a flat uint16 token file."""

    def __init__(self, path: str, seq_len: int, seed: int = 0):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- prepare the dataset first")
        self.path = path
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self._data: Optional[np.memmap] = None

    @property
    def data(self) -> np.memmap:
        # re-opened lazily so the memmap is never pickled into worker processes
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self) -> int:
        return len(self.data)

    def batch(
        self, batch_size: int, device: torch.device, deterministic_offset: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data
        high = len(data) - self.seq_len - 1
        if deterministic_offset is None:
            idx = self.rng.integers(0, high, size=batch_size)
        else:
            idx = (deterministic_offset + np.arange(batch_size) * (self.seq_len + 1)) % high
        x = np.stack([data[i : i + self.seq_len].astype(np.int64) for i in idx])
        y = np.stack([data[i + 1 : i + 1 + self.seq_len].astype(np.int64) for i in idx])
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if device.type == "cuda":
            return (
                xt.pin_memory().to(device, non_blocking=True),
                yt.pin_memory().to(device, non_blocking=True),
            )
        return xt.to(device), yt.to(device)


def build_streams(cfg: DataConfig, seed: int = 0) -> Tuple[TokenStream, TokenStream]:
    train = TokenStream(os.path.join(cfg.data_dir, "train.bin"), cfg.seq_len, seed=seed)
    val = TokenStream(os.path.join(cfg.data_dir, "val.bin"), cfg.seq_len, seed=seed + 1)
    return train, val


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the TinyStories token stream")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    cfg: Config = load_config(args.config, overrides)
    meta = prepare(cfg.data, force=args.force)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
