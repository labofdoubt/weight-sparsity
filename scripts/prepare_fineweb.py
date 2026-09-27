"""Tokenize FineWeb-Edu into the repo's flat uint16 bin format.

    python scripts/prepare_fineweb.py --out /workspace/data/fineweb_edu_10bt \
        --name sample-10BT --val-tokens 50000000 --num-proc 32

Produces train.bin / val.bin / meta.json exactly as data.py does for
TinyStories, so build_streams() and every config field work unchanged --
point data.data_dir at the output directory and set data.seq_len.

Split design: the VALIDATION HOLDOUT is the first documents of the sample
until --val-tokens is reached (document-aligned, fixed and reproducible);
everything after goes to train.bin.  The routine validation subset is NOT a
separate file: evaluate() reads deterministic disjoint windows from the start
of val.bin, so train.val_batches (x micro_batch x (seq_len+1) tokens) IS the
routine subset -- ~5M tokens at the intended settings -- and
train.final_val_batches scores the full holdout once at the end of training.

The tokenizer is the repo's gpt_neo (vocab 50257, eos 50256), one EOS after
every document, matching the TinyStories bins.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.tokenizer import load_gpt_neo_tokenizer  # noqa: E402


def write_bin(tokenized, path: str) -> int:
    """Concatenate a tokenized dataset's ``ids`` column into a uint16 memmap."""
    total = int(np.sum(tokenized["len"], dtype=np.int64))
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
    offset = 0
    shards = max(1, min(1024, len(tokenized) // 10000))
    for shard in range(shards):
        batch = tokenized.shard(num_shards=shards, index=shard, contiguous=True)
        flat = np.concatenate([np.asarray(x, dtype=np.uint16) for x in batch["ids"]])
        arr[offset:offset + len(flat)] = flat
        offset += len(flat)
    arr.flush()
    del arr
    assert offset == total, (offset, total)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--val-tokens", type=int, default=50_000_000)
    ap.add_argument("--num-proc", type=int, default=32)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-docs", type=int, default=0,
                    help="cap the document count (smoke tests only)")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    tok = load_gpt_neo_tokenizer()
    assert tok.vocab_size < 2 ** 16

    ds = load_dataset(args.dataset, name=args.name, split="train",
                      cache_dir=args.cache_dir, num_proc=max(1, args.num_proc))
    if args.max_docs:
        ds = ds.select(range(min(args.max_docs, len(ds))))
    print(f"[fineweb] {args.dataset}:{args.name}: {len(ds):,} documents")

    def tokenize(batch):
        ids = [tok.encode(t, add_eos=True) for t in batch["text"]]
        return {"ids": ids, "len": [len(i) for i in ids]}

    tokenized = ds.map(tokenize, batched=True, batch_size=1000,
                       remove_columns=ds.column_names,
                       num_proc=max(1, args.num_proc), desc="tokenizing")

    # document-aligned split: the first documents form the fixed holdout
    lens = np.asarray(tokenized["len"], dtype=np.int64)
    cum = np.cumsum(lens)
    n_val_docs = int(np.searchsorted(cum, args.val_tokens, side="left")) + 1
    n_val_docs = min(n_val_docs, len(tokenized) - 1)
    val_ds = tokenized.select(range(n_val_docs))
    train_ds = tokenized.select(range(n_val_docs, len(tokenized)))
    print(f"[fineweb] holdout: first {n_val_docs:,} documents "
          f"({int(cum[n_val_docs - 1]):,} tokens; requested {args.val_tokens:,})")

    counts = {}
    for split_name, split in (("val", val_ds), ("train", train_ds)):
        counts[split_name] = write_bin(split, os.path.join(args.out, f"{split_name}.bin"))
        print(f"[fineweb] wrote {split_name}.bin: {counts[split_name]:,} tokens")

    meta = {
        "dataset": f"{args.dataset}:{args.name}",
        "tokenizer": "gpt_neo",
        "vocab_size": int(tok.vocab_size),
        "eos_id": int(tok.eos_id),
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fineweb] meta.json: {meta}")


if __name__ == "__main__":
    main()
