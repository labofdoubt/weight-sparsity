"""Tokenizers for TinyStories.

Two options:

``gpt_neo`` (default)
    The GPT-Neo tokenizer (``EleutherAI/gpt-neo-125M``), i.e. the GPT-2
    byte-level BPE with 50257 tokens.  This is what the original TinyStories
    models were trained with, so it is the "typical" choice.

``bpe``
    A small byte-level BPE trained on TinyStories itself (default 8192
    tokens).  TinyStories has a tiny vocabulary, so this keeps the embedding
    matrix small and lets almost all parameters live in the transformer body.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from .config import DataConfig

EOS_TEXT = "<|endoftext|>"


class Tokenizer:
    """Minimal interface used by the rest of the code base."""

    def __init__(self, impl, vocab_size: int, eos_id: int, kind: str):
        self._impl = impl
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.kind = kind

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        if self.kind == "gpt_neo":
            ids = self._impl(text, add_special_tokens=False)["input_ids"]
        else:
            ids = self._impl.encode(text).ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        ids = [int(i) for i in ids]
        if self.kind == "gpt_neo":
            return self._impl.decode(ids)
        return self._impl.decode(ids)


def load_gpt_neo_tokenizer() -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the gpt_neo tokenizer needs `transformers` (pip install transformers)"
        ) from exc
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids(EOS_TEXT)
    # GPT-Neo's tokenizer reports len() == 50257 including <|endoftext|>
    return Tokenizer(tok, vocab_size=len(tok), eos_id=int(eos_id), kind="gpt_neo")


def load_bpe_tokenizer(path: str) -> Tokenizer:
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the bpe tokenizer needs `tokenizers` (pip install tokenizers)") from exc
    file = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"no trained BPE tokenizer at {file}; run `python -m wsparse.data prepare` first"
        )
    impl = HFTokenizer.from_file(file)
    eos_id = impl.token_to_id(EOS_TEXT)
    return Tokenizer(impl, vocab_size=impl.get_vocab_size(), eos_id=int(eos_id), kind="bpe")


def train_bpe_tokenizer(
    texts, path: str, vocab_size: int = 8192, max_docs: Optional[int] = None
) -> Tokenizer:
    """Train a byte-level BPE on an iterable of strings and save it."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers

    impl = HFTokenizer(models.BPE(unk_token=None))
    impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    impl.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def iterator():
        for i, t in enumerate(texts):
            if max_docs is not None and i >= max_docs:
                break
            yield t

    impl.train_from_iterator(iterator(), trainer=trainer)
    os.makedirs(path, exist_ok=True)
    impl.save(os.path.join(path, "tokenizer.json"))
    return Tokenizer(
        impl, vocab_size=impl.get_vocab_size(), eos_id=int(impl.token_to_id(EOS_TEXT)), kind="bpe"
    )


def build_tokenizer(cfg: DataConfig) -> Tokenizer:
    if cfg.tokenizer == "gpt_neo":
        return load_gpt_neo_tokenizer()
    if cfg.tokenizer == "bpe":
        return load_bpe_tokenizer(cfg.tokenizer_path)
    raise ValueError(f"unknown tokenizer: {cfg.tokenizer}")
