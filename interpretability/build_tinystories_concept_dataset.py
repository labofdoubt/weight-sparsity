"""Build the frozen token-anchored concept dataset (spec sections 3-10).

Reads the *held-out* token stream (``val.bin``) that the models were never
trained on, recovers the individual stories from it, and emits balanced
binary-labelled examples anchored at single token positions.

Working from ``val.bin`` rather than re-downloading TinyStories is deliberate:
the token ids are then bit-identical to what the model consumed, so a target
token index means exactly the same thing here and at activation-extraction
time.  ``stories_index.json`` pins the byte offsets and records a digest of
``val.bin`` so drift is detected rather than silently tolerated.

    python -m interpretability.build_tinystories_concept_dataset \
        --data-dir /workspace/data/tinystories --out benchmark_data

The output is frozen and shared by every model configuration; it must not be
regenerated per model.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wsparse.config import DataConfig  # noqa: E402
from wsparse.data import load_meta  # noqa: E402
from wsparse.tokenizer import build_tokenizer  # noqa: E402

# spec section 8
SPLIT_CAPS = {"train": 500, "validation": 100, "test": 250}
SPLIT_MINIMUMS = {"train": 150, "validation": 50, "test": 100}
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
OPEN_QUOTES, CLOSE_QUOTES, FLAT_QUOTES = "“", "”", '"'
PRONOUN_CONCEPTS = {"male_pronoun", "female_pronoun", "first_person_pronoun"}


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def split_stories(tokens: np.ndarray, eos_id: int, min_len: int, max_len: int):
    """Yield ``(start, end)`` offsets of EOS-delimited stories in the stream."""
    breaks = np.flatnonzero(tokens == eos_id)
    start = 0
    for b in breaks:
        if min_len <= b - start <= max_len:
            yield int(start), int(b)  # the EOS itself is not part of the story
        start = int(b) + 1


def bytes_to_unicode() -> Dict[int, str]:
    """The GPT-2 byte<->unicode table (reproduced rather than imported)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs, n = bs[:], 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def token_bytes_table(tokenizer, vocab_size: int) -> List[bytes]:
    """Raw byte string for every token id.

    Decoding tokens one at a time is wrong for byte-level BPE: a multi-byte
    character can straddle two tokens, and each half decodes to U+FFFD.  That
    silently destroyed curly quotation marks, which the dialogue concept is
    defined by.  Concatenating *bytes* and decoding once is exact.
    """
    impl = getattr(tokenizer, "_impl", None)
    decoder = {v: k for k, v in bytes_to_unicode().items()}
    try:
        if tokenizer.kind == "gpt_neo":
            pieces = impl.convert_ids_to_tokens(list(range(vocab_size)))
        else:
            inv = {i: t for t, i in impl.get_vocab().items()}
            pieces = [inv.get(i, "") for i in range(vocab_size)]
        table = []
        for piece in pieces:
            table.append(bytes(decoder[c] for c in piece))
        return table
    except (AttributeError, KeyError, TypeError) as exc:
        print(f"[build] byte table unavailable ({exc}); falling back to per-token decode")
        return [tokenizer.decode([i]).encode("utf-8") for i in range(vocab_size)]


# TinyStories carries a well-known mojibake artefact (UTF-8 punctuation read as
# Latin-1), which turns curly quotes into sequences like "a<euro><oe>".  Those
# stories cannot be quote-parsed reliably, so they are dropped rather than
# mislabelled.
# UTF-8 punctuation (E2 80 xx) read back as CP1252 -- the curly-quote case.
MOJIBAKE = re.compile("\u00e2\u20ac|\u00e2\u0080|\u00c3[\u00a0-\u00bf]")


def quote_mask(text: str) -> np.ndarray:
    """``True`` for characters lying inside a quotation, excluding the marks."""
    inside = np.zeros(len(text), dtype=bool)
    state = False
    for i, ch in enumerate(text):
        if ch == OPEN_QUOTES:
            state = True
        elif ch == CLOSE_QUOTES:
            state = False
        elif ch == FLAT_QUOTES:
            state = not state
        else:
            inside[i] = state
    return inside


class Story:
    """One held-out story, with the word <-> token alignment already resolved."""

    __slots__ = ("story_id", "start", "end", "text", "words")

    def __init__(self, story_id, start, end, token_ids, table):
        self.story_id, self.start, self.end = story_id, start, end
        pieces = [table[t] for t in token_ids]
        blob = b"".join(pieces)
        self.text = blob.decode("utf-8", errors="replace")

        # byte spans -> character spans, so a multi-byte character that spans
        # two tokens is attributed to the token that completes it
        byte_to_char = np.zeros(len(blob) + 1, dtype=np.int64)
        bi = 0
        for ci, ch in enumerate(self.text):
            n = len(ch.encode("utf-8"))
            byte_to_char[bi : bi + n] = ci
            bi += n
        byte_to_char[bi:] = len(self.text)

        starts, ends, pos = [], [], 0
        for p in pieces:
            starts.append(byte_to_char[pos])
            pos += len(p)
            ends.append(byte_to_char[pos])
        starts, ends = np.array(starts), np.array(ends)
        inside = quote_mask(self.text)

        self.words = []
        for m in WORD_RE.finditer(self.text):
            a, b = m.span()
            hit = np.flatnonzero((starts < b) & (ends > a))
            if hit.size == 0:
                continue
            # the causal model has seen the whole word only at its last subtoken
            self.words.append(
                dict(
                    word=m.group(0).lower().replace("’", "'"),
                    target_token_index=int(hit[-1]),
                    n_subtokens=int(hit.size),
                    char_start=a,
                    char_end=b,
                    in_quote=bool(inside[a:b].all()),
                )
            )


# --------------------------------------------------------------------------- #
# concepts
# --------------------------------------------------------------------------- #


def resolve_concepts(spec: Dict, present: set) -> Tuple[Dict, Dict]:
    """Expand lexicon references and drop words absent from the corpus.

    A word in both pools of the same concept is ambiguous and is dropped from
    that concept rather than forced into a class (spec section 7).
    """
    for key, words in spec["lexicons"].items():
        bad = [w for w in words if not isinstance(w, str)]
        if bad:
            # YAML 1.1 turns bare no/yes/on/off into booleans; quote them
            raise ValueError(
                f"lexicon {key!r} has non-string entries {bad!r} -- quote them in the YAML"
            )
    lex = {k: [w.lower() for w in v] for k, v in spec["lexicons"].items()}
    resolved, report = {}, {}
    for name, cfg in spec["concepts"].items():
        if cfg.get("kind") == "contextual":
            resolved[name] = dict(kind="contextual", group=cfg["group"])
            report[name] = dict(kind="contextual")
            continue
        pos = {w for key in cfg["positive"] for w in lex[key]}
        neg = {w for key in cfg["negative"] for w in lex[key]}
        ambiguous = sorted(pos & neg)
        pos, neg = pos - set(ambiguous), neg - set(ambiguous)
        missing_pos = sorted(w for w in pos if w not in present)
        missing_neg = sorted(w for w in neg if w not in present)
        resolved[name] = dict(
            kind="lexical",
            group=cfg["group"],
            positive=sorted(pos & present),
            negative=sorted(neg & present),
        )
        report[name] = dict(
            kind="lexical",
            ambiguous_dropped=ambiguous,
            absent_from_corpus_positive=missing_pos,
            absent_from_corpus_negative=missing_neg,
        )
    return resolved, report


def collect_lexical(stories, concept, split_of):
    pos, neg = set(concept["positive"]), set(concept["negative"])
    rows = []
    for st in stories:
        for w in st.words:
            label = 1 if w["word"] in pos else (0 if w["word"] in neg else None)
            if label is None:
                continue
            rows.append(_row(st, w, label, split_of[st.story_id]))
    return rows


def collect_dialogue(stories, split_of, rng, min_len=3):
    """Positives inside quotes, negatives the *same words* outside them.

    Matching on lexical identity is the whole point: without it the task
    degenerates into recognizing words that happen to occur in speech.
    """
    by_word = defaultdict(lambda: defaultdict(list))
    for st in stories:
        for w in st.words:
            if len(w["word"]) < min_len:
                continue
            by_word[w["word"]][w["in_quote"]].append((st, w))
    rows = []
    for word, sides in by_word.items():
        inside, outside = sides.get(True, []), sides.get(False, [])
        if not inside or not outside:
            continue  # not lexically matched -- would test word identity
        n = min(len(inside), len(outside))
        for pool, label in ((rng.sample(inside, n), 1), (rng.sample(outside, n), 0)):
            for st, w in pool:
                rows.append(_row(st, w, label, split_of[st.story_id]))
    return rows


def _row(story, word, label, split):
    a = max(0, word["char_start"] - 90)
    b = min(len(story.text), word["char_end"] + 90)
    return dict(
        story_id=story.story_id,
        split=split,
        label=label,
        target_word=word["word"],
        target_token_index=word["target_token_index"],
        n_subtokens=word["n_subtokens"],
        single_token=word["n_subtokens"] == 1,
        context=story.text[a:b].replace("\n", " "),
        char_start=word["char_start"],
        char_end=word["char_end"],
    )


def balance(rows, concept_name, rng, prefer_single_token=True):
    """Downsample to a balanced, capped set per split; never duplicate."""
    out, counts = [], {}
    for split, cap in SPLIT_CAPS.items():
        here = [r for r in rows if r["split"] == split]
        pos = [r for r in here if r["label"] == 1]
        neg = [r for r in here if r["label"] == 0]
        if prefer_single_token:
            # section 4: prefer clean one-token words, fall back rather than
            # lose the concept
            pos_s = [r for r in pos if r["single_token"]]
            neg_s = [r for r in neg if r["single_token"]]
            if min(len(pos_s), len(neg_s)) >= SPLIT_MINIMUMS[split]:
                pos, neg = pos_s, neg_s
        n = min(cap, len(pos), len(neg))
        counts[split] = n
        if n:
            out.extend(rng.sample(pos, n))
            out.extend(rng.sample(neg, n))
    meets = all(counts[s] >= SPLIT_MINIMUMS[s] for s in SPLIT_CAPS)
    return out, counts, meets


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def diversity(df, concept, name):
    def side(label):
        w = df[df.label == label].target_word
        return dict(
            n=int(len(w)),
            unique_words=int(w.nunique()),
            top20=[[k, int(v)] for k, v in Counter(w).most_common(20)],
        )

    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    exempt = name in PRONOUN_CONCEPTS
    return dict(
        group=concept["group"],
        kind=concept["kind"],
        positive=side(1),
        negative=side(0),
        unique_positive_train=int(tr[tr.label == 1].target_word.nunique()),
        unique_positive_test=int(te[te.label == 1].target_word.nunique()),
        single_token_fraction=float(df.single_token.mean()) if len(df) else 0.0,
        diversity_ok=bool(
            exempt
            or (
                tr[tr.label == 1].target_word.nunique() >= 5
                and te[te.label == 1].target_word.nunique() >= 3
            )
        ),
        diversity_exempt=exempt,
    )


def write_qc(path, frames, rng, n=20):
    parts = [
        "<meta charset='utf-8'><style>body{font:14px/1.5 -apple-system,sans-serif;"
        "margin:2rem;max-width:60rem}h2{margin-top:2rem;border-bottom:1px solid #ccc}"
        "mark{background:#ffe08a;font-weight:600}td{padding:.2rem .6rem;vertical-align:top}"
        ".p{color:#137333}.n{color:#a50e0e}</style><h1>Concept QC sample</h1>"
        "<p>20 random positive and 20 random negative examples per concept. "
        "The target token's word is highlighted.</p>"
    ]
    for name, df in frames.items():
        parts.append(f"<h2>{html.escape(name)}</h2><table>")
        for label, cls in ((1, "p"), (0, "n")):
            sub = df[df.label == label]
            take = sub.sample(min(n, len(sub)), random_state=rng) if len(sub) else sub
            for _, r in take.iterrows():
                ctx = html.escape(r.context)
                word = html.escape(r.target_word)
                ctx = re.sub(
                    f"({re.escape(word)})", r"<mark>\1</mark>", ctx, count=1, flags=re.I
                )
                parts.append(
                    f"<tr><td class='{cls}'>{label}</td>"
                    f"<td><b>{word}</b></td><td>{ctx}</td></tr>"
                )
        parts.append("</table>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="directory holding val.bin/meta.json")
    ap.add_argument("--out", default="benchmark_data")
    ap.add_argument("--concepts", default=os.path.join(os.path.dirname(__file__), "concepts.yaml"))
    ap.add_argument("--max-stories", type=int, default=20000)
    ap.add_argument("--min-story-tokens", type=int, default=40)
    ap.add_argument("--max-story-tokens", type=int, default=480)
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.concepts))
    rng = random.Random(spec["seed"])
    os.makedirs(args.out, exist_ok=True)

    meta = load_meta(args.data_dir)
    val_path = os.path.join(args.data_dir, "val.bin")
    digest = hashlib.sha256(open(val_path, "rb").read(1 << 22)).hexdigest()[:16]
    tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"[build] {val_path}: {len(tokens):,} held-out tokens (digest {digest})")

    tokenizer = build_tokenizer(
        DataConfig(data_dir=args.data_dir, tokenizer=meta["tokenizer"])
    )
    table = token_bytes_table(tokenizer, meta["vocab_size"])

    spans = list(
        split_stories(tokens, meta["eos_id"], args.min_story_tokens, args.max_story_tokens)
    )[: args.max_stories]
    print(f"[build] recovered {len(spans):,} stories")

    # --- story-level split, before any example is extracted ---------------- #
    ids = list(range(len(spans)))
    rng.shuffle(ids)
    n_tr = int(SPLIT_FRACTIONS["train"] * len(ids))
    n_va = int(SPLIT_FRACTIONS["validation"] * len(ids))
    split_ids = {
        "train": sorted(ids[:n_tr]),
        "validation": sorted(ids[n_tr : n_tr + n_va]),
        "test": sorted(ids[n_tr + n_va :]),
    }
    split_of = {i: s for s, group in split_ids.items() for i in group}
    assert not (set(split_ids["train"]) & set(split_ids["validation"]))
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    assert not (set(split_ids["validation"]) & set(split_ids["test"]))

    stories, dropped = [], 0
    for i, (s, e) in enumerate(spans):
        story = Story(i, s, e, np.asarray(tokens[s:e], dtype=np.int64), table)
        if MOJIBAKE.search(story.text) or "\ufffd" in story.text:
            dropped += 1
            continue
        stories.append(story)
    print(f"[build] dropped {dropped:,} stories with mojibake/undecodable text")
    present = {w["word"] for st in stories for w in st.words}
    print(f"[build] {len(present):,} distinct word forms in the held-out corpus")

    concepts, vocab_report = resolve_concepts(spec, present)

    frames, stats = {}, {}
    for name, concept in concepts.items():
        if concept["kind"] == "contextual":
            rows = collect_dialogue(stories, split_of, rng)
        else:
            rows = collect_lexical(stories, concept, split_of)
        kept, counts, meets = balance(rows, name, rng)
        if not kept:
            print(f"[build] {name:22} no examples -- skipped")
            continue
        df = pd.DataFrame(kept)
        df.insert(0, "concept", name)
        frames[name] = df
        info = diversity(df, concept, name)
        info.update(
            per_split=counts,
            meets_minimum=bool(meets),
            core=bool(meets and info["diversity_ok"]),
            vocabulary=vocab_report[name],
        )
        stats[name] = info
        flag = "core" if info["core"] else "EXCLUDED from core aggregate"
        print(
            f"[build] {name:22} train/val/test = "
            f"{counts['train']:>4}/{counts['validation']:>4}/{counts['test']:>4} per class"
            f" | {info['positive']['unique_words']:>3} pos word forms | {flag}"
        )

    everything = pd.concat(frames.values(), ignore_index=True)
    for split in SPLIT_CAPS:
        part = everything[everything.split == split]
        part.to_parquet(os.path.join(args.out, f"{split}.parquet"), index=False)

    json.dump(
        {
            "val_bin_digest": digest,
            "n_stories": len(spans),
            "story_spans": {str(i): list(s) for i, s in enumerate(spans)},
            "splits": split_ids,
        },
        open(os.path.join(args.out, "split_story_ids.json"), "w"),
    )
    json.dump(
        {
            "version": spec["version"],
            "seed": spec["seed"],
            "tokenizer": meta["tokenizer"],
            "n_stories": len(spans),
            "n_examples": int(len(everything)),
            "n_concepts": len(stats),
            "n_core_concepts": sum(1 for v in stats.values() if v["core"]),
            "split_caps": SPLIT_CAPS,
            "split_minimums": SPLIT_MINIMUMS,
            "concepts": stats,
        },
        open(os.path.join(args.out, "dataset_stats.json"), "w"),
        indent=2,
    )
    write_qc(os.path.join(args.out, "qc_examples.html"), frames, spec["seed"])

    core = sum(1 for v in stats.values() if v["core"])
    print(
        f"[build] wrote {len(everything):,} examples over {len(stats)} concepts "
        f"({core} core) to {args.out}/"
    )


if __name__ == "__main__":
    main()
