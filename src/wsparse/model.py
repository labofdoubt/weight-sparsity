"""A small decoder-only transformer for TinyStories.

Defaults follow the spec: no biases, RMSNorm, learnable absolute positional
embeddings, pre-norm residual blocks, fused QKV projection and SDPA attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.attn_dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Standard 2-layer MLP, or SwiGLU when ``mlp_activation == "swiglu"``.

    For SwiGLU ``fc1`` produces both the gate and the value branch, so the
    layer keeps the same parameter count as the plain variant for a given
    ``mlp_ratio``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.activation = cfg.mlp_activation
        d_mlp = cfg.d_mlp
        fan_out = 2 * d_mlp if self.activation == "swiglu" else d_mlp
        self.fc1 = nn.Linear(cfg.d_model, fan_out, bias=cfg.bias)
        self.fc2 = nn.Linear(d_mlp, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if self.activation == "swiglu":
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        elif self.activation == "gelu":
            h = F.gelu(h, approximate="tanh")
        elif self.activation == "relu":
            h = F.relu(h)
        else:  # silu
            h = F.silu(h)
        return self.dropout(self.fc2(h))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)
        # Two insertion points for the activation bottleneck, and the choice
        # matters more than it looks.  `mlp_bottleneck` wraps the tensor fed to
        # the MLP, so it sits *inside* a residual branch and the skip routes
        # around it -- whatever it discards is still carried forward by x.
        # `residual_bottleneck` replaces the stream itself before attention, so
        # nothing routes around it: every later op in the block, and every later
        # block, sees only what survived.  Identity by default, and Identity
        # holds no parameters or buffers, so state_dicts and parameter counts
        # are unchanged when the experiment is off.
        self.residual_bottleneck = nn.Identity()
        self.mlp_bottleneck = nn.Identity()
        # `residual_out` is the same stream, at the far end of the block.  Note
        # that with every layer selected the two stream placements differ in
        # only one position out of n_layers + 1: `residual` also bottlenecks the
        # embedding output before block 0 but never the final hidden state,
        # while `residual_out` never touches the embedding but does bottleneck
        # the state that feeds norm_f and the unembedding.  The n_layers - 1
        # interior positions are identical.
        self.residual_out_bottleneck = nn.Identity()
        # `post_attn` / `post_mlp` sit on a sub-block's *output*, before it is
        # added back to the stream.  Like `pre_mlp` they live inside a residual
        # branch, so the skip still carries x -- but they constrain what the
        # branch may contribute rather than what it may read.  Both can be
        # active at once, and each gets its own parameters.
        self.post_attn_bottleneck = nn.Identity()
        self.post_mlp_bottleneck = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_bottleneck(x)
        x = x + self.post_attn_bottleneck(self.attn(self.norm1(x)))
        x = x + self.post_mlp_bottleneck(self.mlp(self.mlp_bottleneck(self.norm2(x))))
        x = self.residual_out_bottleneck(x)
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)  # learnable
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        if cfg.init_scale_residual:
            scale = 1.0 / math.sqrt(2 * cfg.n_layers)
            for block in self.blocks:
                block.attn.proj.weight.data.mul_(scale)
                block.mlp.fc2.weight.data.mul_(scale)

    # ---- init ------------------------------------------------------------ #
    def _linear_std(self, weight: torch.Tensor) -> float:
        cfg = self.cfg
        if cfg.init_scheme == "fan_in":
            fan_in = weight.shape[1]
            return cfg.init_gain / math.sqrt(fan_in)
        return cfg.init_std

    def _init_weights(self, module: nn.Module) -> None:
        cfg = self.cfg
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self._linear_std(module.weight))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = cfg.init_std_embedding if cfg.init_std_embedding is not None else cfg.init_std
            if module is getattr(self, "pos_emb", None) and cfg.init_std_pos is not None:
                std = cfg.init_std_pos
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ---- bookkeeping ------------------------------------------------------ #
    def _mask_parameter_ids(self) -> set:
        """ids of the auxiliary sparsity parameters, if any layer is masked.

        Duck-typed on ``mask_parameters`` to avoid importing ``wsparse.sparsity``
        here (that module imports this one).
        """
        ids = set()
        for module in self.modules():
            fn = getattr(module, "mask_parameters", None)
            if callable(fn):
                ids.update(id(p) for p in fn())
        return ids

    def num_parameters(self, non_embedding: bool = False, include_mask: bool = False) -> int:
        """Count model parameters.

        By default the sparsity parameters (LTP thresholds / CS gates) are
        excluded, so the number stays comparable to the dense model.
        """
        skip = set() if include_mask else self._mask_parameter_ids()
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen or id(p) in skip:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
            if self.cfg.tie_embeddings is False:
                total -= self.lm_head.weight.numel()
        return total

    # ---- forward ---------------------------------------------------------- #
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.emb_dropout(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``generator`` makes sampling reproducible independently of how much
        global RNG the training loop happens to have consumed."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def build_model(cfg: ModelConfig) -> TransformerLM:
    return TransformerLM(cfg)
