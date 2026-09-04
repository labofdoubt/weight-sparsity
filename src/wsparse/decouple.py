"""Magnitude-direction decoupled optimization (arXiv:2606.25971).

Every matrix weight is treated as a fixed-norm *direction* times learnable
per-row / per-column *magnitude gains*,

    W = diag(g_row) @ W_hat @ diag(g_col),      ||W_hat||_F = c_F,

but the model only ever holds the fused ``W``: the split lives entirely inside
the optimizer step (the paper's Algorithm 2), so the forward/backward pass pays
nothing.  Each step, per matrix:

    1. materialize the positive gains  g = softplus(raw)
    2. recover the direction           W_hat = W / (g_row g_col^T)
    3. split the gradient              G_hat = g_row * G * g_col
                                       g_grow = rowsum(W_hat*G * g_col) * phi'
                                       g_gcol = colsum(g_row * W_hat*G) * phi'
    4. Adam-step the direction, project back:  W_hat <- c_F W_hat / ||W_hat||
    5. Adam-step the raw gains (their own moments, the same group LR)
    6. refuse                          W = diag(g_row') W_hat diag(g_col')

Embeddings and the LM head are the special case: each row is one token's
vector, so rows are held at unit L2 norm (plain Adam on the fused weight, then a
row projection) with **no** gains; the input embedding is upscaled by a fixed
``sqrt(d)`` in the forward instead.  Under ``tie_embeddings`` there is a single
such matrix and one projection covers both roles.

Everything norm-constrained trains **without weight decay** -- the sphere is the
regularizer -- which is why this module never reads the config's weight_decay.

The sphere radius is the initialization norm.  ``md_init_`` initializes every
matrix entrywise ``N(0, 1/d_model)`` and then projects *exactly* onto
``c_F = sqrt(d_out * d_in / d_model)`` (equal to ``sqrt(max(d_out, d_in))``
whenever the smaller dimension is ``d_model``, which holds for every matrix in
this codebase, bottleneck projections included).  ``c_F`` is captured into the
optimizer state on first sight of each parameter and travels with checkpoints.

Two gain placements are supported (``decouple_gains``):

    "row_col"  each matrix gets both g_row and g_col   (the paper's default)
    "up_down"  d_out >= d_in gets g_row only; d_out < d_in gets g_col only
               (the nGPT-style alternation: up-projections scale their new
               rows, down-projections their incoming columns)

Deliberate deviations from the paper's *experimental setup* (not the method):
the base optimizer here is Adam with this project's betas and one shared LR
schedule for every group -- the paper fixes separate embedding/head LRs and
runs warmup-free.  Both are configuration, not code: the gains already follow
the matrix group's LR, and warmup is ``train.warmup_steps=0`` away.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

# softplus(RAW_GAIN_ONE) == 1, so every gain starts exactly at 1 and the fused
# weight equals the direction at initialization.
RAW_GAIN_ONE = math.log(math.e - 1.0)

GAIN_MODES = ("row_col", "up_down")


def _wants_gains(shape: torch.Size, mode: str):
    """``(row, col)`` booleans for one matrix under a gain placement."""
    if mode == "row_col":
        return True, True
    d_out, d_in = shape[0], shape[1]
    return (True, False) if d_out >= d_in else (False, True)


@torch.no_grad()
def md_init_(model, gain_mode: str = "row_col") -> Dict[str, int]:
    """Re-initialize ``model`` in place for magnitude-direction training.

    Overrides *every* other initialization choice -- ``init_scheme`` /
    ``init_std`` / ``init_gain`` / ``init_std_embedding`` /
    ``init_scale_residual`` and the bottleneck's own ``init_mode`` family -- as
    the decouple flag promises.  Walks the finished model (bottlenecks already
    spliced in), so anything added after ``build_model`` is covered too.

    * embeddings and the (untied) LM head: rows drawn Gaussian, projected to
      unit L2 norm;
    * every other dim>=2 weight: entrywise ``N(0, 1/d_model)``, projected to
      exactly ``c_F = sqrt(d_out*d_in/d_model)`` in Frobenius norm;
    * biases and 1-D gains (RMSNorm) are zeroed / left at their own defaults.
    """
    if gain_mode not in GAIN_MODES:
        raise ValueError(f"unknown decouple_gains: {gain_mode!r} ({' | '.join(GAIN_MODES)})")
    d_model = int(model.cfg.d_model)
    embed_ids = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    counts = {"embed": 0, "matrix": 0}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids:
            torch.nn.init.normal_(p, mean=0.0, std=1.0)
            p.div_(p.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            counts["embed"] += 1
        elif p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(d_model))
            c_f = math.sqrt(p.shape[0] * p.shape[1] / d_model)
            p.mul_(c_f / p.norm().clamp_min(1e-12))
            counts["matrix"] += 1
        elif "bias" in name:
            p.zero_()
        # 1-D norm gains keep their own initialization (ones)
    return counts


class DecoupledAdamW(torch.optim.Optimizer):
    """Adam with magnitude-direction decoupling for matrix weights.

    Three parameter kinds, tagged per group:

    ``kind="md"``     fused matrices, stepped by the paper's Algorithm 2.  The
                      raw gains and their Adam moments live in ``self.state``
                      (they are derived quantities of the training method, not
                      model parameters -- checkpoints stay plain fused weights).
    ``kind="embed"``  embeddings / untied head: plain Adam then per-row
                      renormalization to unit L2.
    ``kind="plain"``  everything else (norm gains, biases): plain Adam.

    No parameter kind uses weight decay.  ``lr`` is read from each group at
    step time, so the existing ``set_lr`` schedule drives gains too (the paper
    lets gains follow the matrix LR).
    """

    def __init__(self, param_groups: List[Dict], betas=(0.9, 0.95), eps: float = 1e-8,
                 gain_mode: str = "row_col"):
        if gain_mode not in GAIN_MODES:
            raise ValueError(f"unknown decouple_gains: {gain_mode!r}")
        defaults = dict(lr=0.0, betas=betas, eps=eps, kind="plain",
                        gain_mode=gain_mode, is_mask=False)
        super().__init__(param_groups, defaults)

    # ---- shared Adam kernel ------------------------------------------------ #
    @staticmethod
    def _adam_(value: torch.Tensor, grad: torch.Tensor, state: dict, prefix: str,
               lr: float, beta1: float, beta2: float, eps: float, step: int) -> None:
        m = state[f"{prefix}m"]
        v = state[f"{prefix}v"]
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        denom = (v / bc2).sqrt_().add_(eps)
        value.addcdiv_(m, denom, value=-lr / bc1)

    def _state_for(self, p: torch.Tensor, kind: str, gain_mode: str) -> dict:
        state = self.state[p]
        if state:
            return state
        state["step"] = 0
        state["m"] = torch.zeros_like(p)
        state["v"] = torch.zeros_like(p)
        if kind == "md":
            row, col = _wants_gains(p.shape, gain_mode)
            if row:
                state["raw_grow"] = torch.full((p.shape[0],), RAW_GAIN_ONE,
                                               device=p.device, dtype=p.dtype)
                state["grow_m"] = torch.zeros_like(state["raw_grow"])
                state["grow_v"] = torch.zeros_like(state["raw_grow"])
            if col:
                state["raw_gcol"] = torch.full((p.shape[1],), RAW_GAIN_ONE,
                                               device=p.device, dtype=p.dtype)
                state["gcol_m"] = torch.zeros_like(state["raw_gcol"])
                state["gcol_v"] = torch.zeros_like(state["raw_gcol"])
            # The sphere radius is the *initialization* norm, captured at first
            # sight (gains are exactly 1 then, so ||W|| is ||W_hat||) and kept in
            # the state so resume preserves it.
            state["c_f"] = p.detach().float().norm().clone()
        return state

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901 -- one method, three kinds
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            kind = group["kind"]
            gain_mode = group["gain_mode"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                state = self._state_for(p, kind, gain_mode)
                state["step"] += 1
                t = state["step"]

                if kind == "plain":
                    self._adam_(p, G, state, "", lr, beta1, beta2, eps, t)
                    continue

                if kind == "embed":
                    # Adam in the ambient space, then each token vector back to
                    # the unit sphere (paper section on embeddings).
                    self._adam_(p, G, state, "", lr, beta1, beta2, eps, t)
                    p.div_(p.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                    continue

                # ---- kind == "md": Algorithm 2 ------------------------------ #
                has_row = "raw_grow" in state
                has_col = "raw_gcol" in state
                grow = F.softplus(state["raw_grow"]) if has_row else None
                gcol = F.softplus(state["raw_gcol"]) if has_col else None

                # recover the on-sphere direction from the fused weight, with
                # exactly the gains it was fused with -- no asymmetric guards
                # (the paper traced an instability to precisely such a mismatch)
                w_hat = p.detach().clone()
                if has_row:
                    w_hat.div_(grow.unsqueeze(1))
                if has_col:
                    w_hat.div_(gcol.unsqueeze(0))

                whg = w_hat * G
                if has_row:
                    g_grow = (whg * gcol.unsqueeze(0)).sum(dim=1) if has_col \
                        else whg.sum(dim=1)
                    g_grow = g_grow * torch.sigmoid(state["raw_grow"])   # phi'
                if has_col:
                    g_gcol = (grow.unsqueeze(1) * whg).sum(dim=0) if has_row \
                        else whg.sum(dim=0)
                    g_gcol = g_gcol * torch.sigmoid(state["raw_gcol"])

                g_hat = G.clone()
                if has_row:
                    g_hat.mul_(grow.unsqueeze(1))
                if has_col:
                    g_hat.mul_(gcol.unsqueeze(0))

                self._adam_(w_hat, g_hat, state, "", lr, beta1, beta2, eps, t)
                w_hat.mul_(state["c_f"] / w_hat.norm().clamp_min(1e-12))

                if has_row:
                    self._adam_(state["raw_grow"], g_grow, state, "grow_",
                                lr, beta1, beta2, eps, t)
                if has_col:
                    self._adam_(state["raw_gcol"], g_gcol, state, "gcol_",
                                lr, beta1, beta2, eps, t)

                # refuse with the *updated* gains
                fused = w_hat
                if has_row:
                    fused = fused * F.softplus(state["raw_grow"]).unsqueeze(1)
                if has_col:
                    fused = fused * F.softplus(state["raw_gcol"]).unsqueeze(0)
                p.copy_(fused)
        return loss


def build_decoupled_optimizer(model, train_cfg, gain_mode: str = "row_col",
                              mask_param_ids: Optional[Iterable[int]] = None):
    """Group the model's parameters for :class:`DecoupledAdamW`.

    Embeddings (and the untied head) are the unit-row kind; every other dim>=2
    weight is a decoupled matrix; the rest (RMSNorm gains, biases) are plain
    Adam.  Weight decay is deliberately absent everywhere -- see the module
    docstring.
    """
    if mask_param_ids:
        raise ValueError(
            "decouple=True with sparsity mask parameters is not supported: the "
            "masked weights' gradients are not the fused-matrix gradients "
            "Algorithm 2 expects"
        )
    embed_ids = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    md, embed, plain, seen = [], [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids:
            embed.append(p)
        elif p.dim() >= 2:
            md.append(p)
        else:
            plain.append(p)
    groups = [
        dict(params=md, kind="md", name="md_matrix", weight_decay=0.0, is_mask=False),
        dict(params=embed, kind="embed", name="md_embed", weight_decay=0.0, is_mask=False),
        dict(params=plain, kind="plain", name="nodecay", weight_decay=0.0, is_mask=False),
    ]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["lr"] = train_cfg.lr
    return DecoupledAdamW(groups, betas=tuple(train_cfg.betas), eps=train_cfg.eps,
                          gain_mode=gain_mode)
