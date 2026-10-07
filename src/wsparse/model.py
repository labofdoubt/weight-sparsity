"""A small decoder-only transformer for TinyStories.

Defaults follow the spec: no biases, RMSNorm, learnable absolute positional
embeddings, pre-norm residual blocks, fused QKV projection and SDPA attention.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import replace
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_STD_EMBEDDING, ModelConfig


class LossDetails(NamedTuple):
    """``TransformerLM.forward(..., return_loss_details=True)``.

    The opt-in interface of the laplace_policy trainer: the unreduced CE and
    per-sequence costs the likelihood-ratio loss needs, plus the policy
    density terms the gates recorded during THIS forward.  A tuple, so a DDP
    wrapper that traverses outputs sees every graph-bearing tensor.

    ``ce`` is ``sum_bt ell_bt / N_valid`` (the ordinary reduction), ``ce_tokens``
    the float32 ``[B, T]`` token CE with ignored targets at 0, ``valid`` their
    mask, ``seq_ce`` each sequence's mean CE (``[B]``, 0 where it has no valid
    target), ``seq_valid`` its valid-target count.  ``policy_log_prob`` is the
    gamma-weighted log density summed over every sampled gate instance and
    position of each sequence (``[B]``, differentiable), ``None`` when no gate
    sampled or grad was disabled; ``policy_records`` are the per-invocation
    records (detached diagnostics; the log-density graphs live in them too,
    so drop the result after ``backward``).
    """

    logits: torch.Tensor
    ce: torch.Tensor
    ce_tokens: torch.Tensor
    valid: torch.Tensor
    seq_ce: torch.Tensor
    seq_valid: torch.Tensor
    policy_log_prob: Optional[torch.Tensor]
    policy_records: List


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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``(B, n_heads, T, head_dim)`` by position (rotate-half convention).

    Each pair ``(x[d], x[d + head_dim/2])`` is treated as a complex number and
    rotated by the position-dependent angle baked into ``cos``/``sin``, so
    ``q_i . k_j`` afterwards depends on positions only through ``i - j``.  A
    rotation, so norms are preserved exactly -- both properties are tested.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.attn_dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = cfg.pos_encoding == "rope"
        if self.rope:
            # The cos/sin cache for every position up to max_seq_len, shaped
            # (1, 1, T, head_dim) with the half-table duplicated so apply_rope
            # needs no reshuffling.  Computed in float32 and cast at use; a
            # *non-persistent* buffer, so state_dicts are identical to the
            # learned-embedding layout minus pos_emb and old checkpoints are
            # unaffected.
            half = self.head_dim // 2
            inv_freq = cfg.rope_theta ** (
                -torch.arange(0, half, dtype=torch.float32) / half
            )
            freqs = torch.outer(
                torch.arange(cfg.max_seq_len, dtype=torch.float32), inv_freq
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("rope_cos", emb.cos()[None, None], persistent=False)
            self.register_buffer("rope_sin", emb.sin()[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            cos = self.rope_cos[..., :T, :].to(q.dtype)
            sin = self.rope_sin[..., :T, :].to(q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
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
        x = self.body(x)
        x = self.residual_out_bottleneck(x)
        return x

    def body(self, x: torch.Tensor) -> torch.Tensor:
        """``forward`` between the two stream placements: ``x + Delta``, with
        the attention and MLP contributions added in the order ``forward`` adds
        them (so the code-residual stack's block 0 is bit-identical to it)."""
        x = x + self.post_attn_bottleneck(self.attn(self.norm1(x)))
        x = x + self.post_mlp_bottleneck(self.mlp(self.mlp_bottleneck(self.norm2(x))))
        return x

    def branches(self, x: torch.Tensor) -> torch.Tensor:
        """The block's residual contribution ``Delta`` with no stream bottleneck.

        ``body`` minus its input: attention then MLP, each computed as in
        ``forward``, returned as the sum of the two contributions rather than
        as ``x + Delta`` (the code-residual stack encodes ``Delta`` alone).
        """
        a = self.post_attn_bottleneck(self.attn(self.norm1(x)))
        m = self.post_mlp_bottleneck(
            self.mlp(self.mlp_bottleneck(self.norm2(x + a))))
        return a + m


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Under rope the position information lives in attention, so there is no
        # table here at all -- None rather than an unused parameter, so it is
        # neither trained, decayed, nor counted.
        self.pos_emb = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model)  # learnable, absolute
            if cfg.pos_encoding == "learned" else None
        )
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # norm_f gives the head a unit-RMS input, so the init logits land at std
        # head_std * sqrt(d_model); divide that out to normalize them to ~1.  A
        # constant, so it stays out of the state_dict and checkpoints load.
        self.logit_mult = 1.0
        if cfg.logit_scale == "auto":
            self.logit_mult = 1.0 / (self._head_std() * math.sqrt(cfg.d_model))
        # Under magnitude-direction decoupling the embedding rows are unit-norm
        # (component std 1/sqrt(d)), so a fixed sqrt(d) upscale puts the
        # residual stream at unit RMS on entry.  A constant, not a parameter.
        # md_init starts from the same unit rows, so it needs the same upscale.
        self.embed_scale = math.sqrt(cfg.d_model) if (cfg.decouple or cfg.md_init) else 1.0

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

    def _embedding_std(self, module: nn.Module) -> float:
        cfg = self.cfg
        if module is getattr(self, "pos_emb", None) and cfg.init_std_pos is not None:
            return cfg.init_std_pos
        if cfg.init_std_embedding is not None:
            return cfg.init_std_embedding
        return DEFAULT_STD_EMBEDDING

    def _head_std(self) -> float:
        """The std ``lm_head.weight`` is initialized with.

        Never routed through ``_linear_std``: the unembedding is scaled from what
        the logits need, not from the linear-layer convention, so ``init_scheme``
        / ``init_std`` / ``init_gain`` do not reach it.
        """
        cfg = self.cfg
        if cfg.tie_embeddings:
            return self._embedding_std(self.tok_emb)  # it *is* the token embedding
        if cfg.init_std_unembedding is not None:
            return cfg.init_std_unembedding
        return 1.0 / math.sqrt(cfg.d_model)  # => init logits at unit std

    def _init_weights(self, module: nn.Module) -> None:
        cfg = self.cfg
        if isinstance(module, nn.Linear):
            # With tied embeddings lm_head.weight *is* tok_emb.weight; leave it to
            # the nn.Embedding branch, which otherwise gets silently overwritten
            # with the linear std (`apply` visits lm_head after tok_emb).
            tied = cfg.tie_embeddings and module.weight is self.tok_emb.weight
            if not tied:
                std = (
                    self._head_std()
                    if module is self.lm_head
                    else self._linear_std(module.weight)
                )
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self._embedding_std(module))

    # ---- bookkeeping ------------------------------------------------------ #
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count model parameters, each shared tensor once."""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                total -= self.pos_emb.weight.numel()
            if self.cfg.tie_embeddings is False:
                total -= self.lm_head.weight.numel()
        return total

    # ---- forward ---------------------------------------------------------- #
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
        return_loss_details: bool = False, policy=None,
    ):
        """``(logits, loss)`` -- or a :class:`LossDetails` with
        ``return_loss_details=True`` (needs ``targets``).

        ``policy`` is an optional ``PolicyForwardSettings`` for the
        laplace_policy gates (sample override, noise generator, collector),
        installed on them for this forward only and restored in ``finally``.
        With ``return_loss_details`` a forward-scoped collector is opened for
        the gates' density terms (unless the settings bring one) and closed
        when the forward ends, so DDP sees the resulting graph among the
        returned tensors and nothing survives the call.  The default two-result
        path is the exact old computation: the unreduced float32 CE is only
        formed in the opt-in branch.
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")
        if return_loss_details and targets is None:
            raise ValueError("return_loss_details=True needs targets")
        x = self.tok_emb(idx)
        if self.embed_scale != 1.0:
            x = x * self.embed_scale
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None]
        x = self.emb_dropout(x)
        collector = None
        ctx = nullcontext()
        if return_loss_details or policy is not None:
            # imported here: wsparse.bottleneck imports this module
            from .bottleneck.laplace_policy import (PolicyCollector, PolicyForwardSettings,
                                                    policy_forward, policy_gates)
            gates = policy_gates(self)
            settings = policy if policy is not None else PolicyForwardSettings()
            if return_loss_details and gates:
                if settings.collector is None:
                    settings = replace(settings, collector=PolicyCollector())
                collector = settings.collector
            ctx = policy_forward(gates, settings)
        try:
            with ctx:
                # set by the bottleneck controller under code_residual; absent
                # otherwise, so plain models keep their state_dict and forward
                if getattr(self, "code_residual", False):
                    x = self._code_residual_stack(x)
                else:
                    for block in self.blocks:
                        x = block(x)
        finally:
            if collector is not None:
                collector.close()
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if self.logit_mult != 1.0:
            logits = logits * self.logit_mult
        if return_loss_details:
            from .bottleneck.laplace_policy import sequence_costs
            flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1),
                ignore_index=-100, reduction="none")
            valid = targets != -100
            ce_tokens = torch.where(valid, flat.view(B, T), flat.new_zeros(()))
            ce, seq_ce, seq_valid = sequence_costs(ce_tokens, valid)
            records = list(collector.records) if collector is not None else []
            log_prob = collector.sequence_log_prob() if collector is not None else None
            return LossDetails(logits, ce, ce_tokens, valid, seq_ce, seq_valid, log_prob, records)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def _code_residual_stack(self, x: torch.Tensor) -> torch.Tensor:
        """The block stack with the K-sparse code, not the stream, carried.

            c_1     = TopK(E_0 (x_0 + Delta_0(x_0))),         x_0 the embedding
            x_l     = norm_l(g_D D_{l-1} c_l),                l = 1 .. L-1
            c_{l+1} = TopK(c_l + alpha * E_l Delta_l(x_l)),   l = 1 .. L-1
            returns g_D D_{L-1} c_L

        Block 0 is the ordinary stream bottleneck: it reads the embedding and
        its bottleneck encodes the block's whole output, so a code-residual
        model and a stream-carried one coincide up to and including ``c_1``.
        From block 1 on, block l reads the code through the previous
        bottleneck's decoder ``D_{l-1}`` (and that module's post-norm, if any),
        writes its contribution ``Delta_l`` through its own encoder ``E_l``,
        and its gate adds the encoded contribution to the carried code instead
        of re-encoding the decoded stream.  Under share_projections every
        ``E_l``, ``D_l`` is one pair.  TopK is the Euclidean projection onto
        K-sparse vectors, so a block with ``Delta = 0`` is the identity on the
        code and the carry's Jacobian is the support mask -- the residual
        connection of a pre-norm transformer, moved into code space.  No module
        is added: the parameters and the state_dict are the stream-carried
        model's.

        Until 2026-10-04 the stack had an entry gate ``c_0 = TopK(E_in x_0)``
        before block 0, block 0 read ``D c_0`` and encoded only its own
        ``Delta_0``; checkpoints from then carry ``code_entry.*`` keys and
        belong to that definition.
        """
        from .bottleneck.rblapsum import CarryChain, carry_scope, split_carry_read
        alpha = float(getattr(self, "code_residual_scale", 1.0))
        # carry scopes (rblapsum_surrogate_scope="carry_*"): every gate's output
        # is split into the copy the next gate carries and the copy the decoder
        # reads, and the gates share one chain per forward for their backward
        chain = None
        if torch.is_grad_enabled() and any(
                carry_scope(getattr(b.residual_out_bottleneck.gate,
                                    "rblapsum_surrogate_scope", "pool"))
                for b in self.blocks):
            chain = CarryChain()
        code = None
        for index, block in enumerate(self.blocks):
            bot = block.residual_out_bottleneck
            if chain is not None:
                bot.gate._carry_chain, bot.gate._carry_index = chain, index
            try:
                if code is None:
                    # block 0: the installed stream bottleneck, up to its gate
                    code = bot.gate(bot.in_proj(block.body(x)))
                else:
                    update = alpha * bot.in_proj(block.branches(x))
                    if getattr(bot.gate, "rblapsum_surrogate_scope", "pool").startswith("update"):
                        code = self._split_surrogate(bot.gate, code, update)
                    else:
                        code = bot.gate(code + update)
            finally:
                if chain is not None:
                    bot.gate._carry_chain, bot.gate._carry_index = None, -1
            read = code
            if chain is not None:
                code, read = split_carry_read(code, chain, index)
            x = bot.post_norm(bot.decode(read))
        return x

    @staticmethod
    def _split_surrogate(gate: nn.Module, carry: torch.Tensor,
                         update: torch.Tensor) -> torch.Tensor:
        """``gate(carry + update)`` with the support term routed to ``update``.

        rblapsum_surrogate_scope="update" (and "update_inactive",
        "update_active"): the forward is the gate's own; the carried code
        receives the exact hard gradient ``m * g`` and the update the gate's
        full gradient (hard + support).  ``y_hard - y_hard.detach()``
        is zero in the forward, so only its gradient ``m * g`` reaches the carry.
        """
        y = gate(carry.detach() + update)
        mask = (y.detach() != 0).to(carry.dtype)
        y_hard = (carry + update.detach()) * mask
        return y + (y_hard - y_hard.detach())

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
