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
from typing import Dict, Iterable, List, Optional, Tuple

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


def _bottleneck_modules(model) -> List:
    """The model's ``SparseTopKBottleneck`` modules, in registration order.

    Imported lazily: ``bottleneck.module`` imports ``model``, which this module
    does not, so a top-level import here would only add a cycle risk for no
    gain.
    """
    from .bottleneck.module import SparseTopKBottleneck

    return [m for m in model.modules() if isinstance(m, SparseTopKBottleneck)]


@torch.no_grad()
def tight_frame_(W: torch.Tensor, d_model: int) -> torch.Tensor:
    """Fill ``W`` with a tight frame over the ``d_model``-sized axis, in place.

    For an encoder ``(d_b, d)`` this gives ``W^T W = (d_b/d) I_d`` (orthonormal
    columns, scaled); for a decoder ``(d, d_b)`` it gives ``W W^T = (d_b/d)
    I_d`` (orthonormal rows, scaled).  ``nn.init.orthogonal_`` already picks the
    right side from the shape -- it orthogonalizes the longer axis -- so the
    only thing to add is the ``sqrt(d_b/d)`` scale.

    That scale is what keeps the two invariants MD relies on: the Frobenius norm
    is ``sqrt(tr((d_b/d) I_d)) = sqrt(d_b)``, which is exactly the sphere radius
    ``c_F = sqrt(d_out d_in / d_model)`` of both bottleneck matrices, and the
    mean squared norm along the ``d_b``-sized axis is 1, so encoder rows and
    decoder columns stay unit-norm on average as under the standard init.
    """
    d_b = W.shape[0] if W.shape[1] == d_model else W.shape[1]
    if min(W.shape) != d_model:
        raise ValueError(
            f"a tight frame over d_model={d_model} needs one axis of that size, "
            f"got {tuple(W.shape)}")
    torch.nn.init.orthogonal_(W)
    W.mul_(math.sqrt(d_b / d_model))
    return W


@torch.no_grad()
def _frame_stats(W: torch.Tensor, d_model: int, singular: bool = True) -> Dict:
    """Tight-frame diagnostics for one bottleneck matrix.

    ``gram_rel_err`` is ``||W^T W - (d_b/d) I||_F / ||(d_b/d) I||_F`` for a tall
    matrix and the ``W W^T`` version for a wide one -- the Gram that a tight
    frame makes isotropic.  ``mean_sq_norm`` averages over the ``d_b``-sized
    axis, so it is the mean squared encoder-row or decoder-column norm.

    Computed in float64 on purpose.  ``md_init_`` runs after ``model.to(device)``
    and after ``allow_tf32 = True``, so an fp32 Gram of a 1536x768 frame reads
    2e-4 on CUDA -- a property of TF32's 10-bit mantissa, not of the frame,
    which is orthonormal to 4e-7.  One float64 matmul per reported matrix (the
    first bottleneck only) is cheap enough to make the number mean what it says.
    """
    Wd = W.detach().double()
    tall = Wd.shape[1] == d_model
    d_b = Wd.shape[0] if tall else Wd.shape[1]
    gram = Wd.t() @ Wd if tall else Wd @ Wd.t()
    target = (d_b / d_model) * torch.eye(d_model, dtype=gram.dtype,
                                         device=gram.device)
    out = {
        "mean_sq_norm": float((Wd ** 2).sum(dim=1 if tall else 0).mean()),
        "frobenius": float(Wd.norm()),
        "gram_rel_err": float((gram - target).norm() / target.norm()),
    }
    if singular:
        sv = torch.linalg.svdvals(Wd)
        out["sv_min"], out["sv_max"] = float(sv.min()), float(sv.max())
    return out


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

    # ---- the bottleneck's own geometry, if it was asked for -------------- #
    bn_init = str(getattr(model.cfg, "bottleneck_init", "standard"))
    scale_mode = str(getattr(model.cfg, "bottleneck_decoder_scale", "none"))
    # Always collected (neither call touches the RNG, so the standard path is
    # bit-identical) so the returned diagnostics are complete in every mode.
    frames, mods = {}, _bottleneck_modules(model)
    if bn_init == "orthogonal":
        for mod in mods:
            if getattr(mod, "tied", False):
                raise ValueError(
                    "bottleneck_init='orthogonal' draws the encoder and the "
                    "decoder as INDEPENDENT frames, which a tied decoder cannot "
                    "be (it is the encoder transposed); set "
                    "activation_bottleneck.tie_encoder_decoder=false")
            # in_proj carries the value branch, score_proj the ranking one under
            # gated_topk -- both map d_model -> n_features, so both are encoders
            for proj in (mod.in_proj, mod.score_proj):
                if proj is not None:
                    frames[id(proj.weight)] = "encoder"
            frames[id(mod.out_proj.weight)] = "decoder"

    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids:
            torch.nn.init.normal_(p, mean=0.0, std=1.0)
            p.div_(p.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            counts["embed"] += 1
        elif id(p) in frames:
            # Same Frobenius norm as the standard branch below (sqrt(d_b) for
            # both bottleneck matrices), so the sphere the optimizer captures is
            # unchanged -- only the direction's spectrum differs.
            tight_frame_(p, d_model)
            counts["matrix"] += 1
        elif p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(d_model))
            c_f = math.sqrt(p.shape[0] * p.shape[1] / d_model)
            p.mul_(c_f / p.norm().clamp_min(1e-12))
            counts["matrix"] += 1
        elif "bias" in name:
            p.zero_()
        # 1-D norm gains keep their own initialization (ones)

    if mods:
        counts.update(_apply_decoder_scale(mods, d_model, scale_mode, bn_init))
    return counts


@torch.no_grad()
def _apply_decoder_scale(mods: List, d_model: int, scale_mode: str,
                         bn_init: str) -> Dict:
    """Set every bottleneck's ``g_D`` and collect the init diagnostics.

    ``g_D = sqrt(d_model / K_eff)`` under ``backward_preserving`` (1 otherwise),
    which is the scale at which an isotropic gradient crosses the bottleneck
    with unit energy: the effective decoder then satisfies ``W_D W_D^T =
    (d_b/K_eff) I`` at init.  It goes nowhere near the row/column gains.

    The expensive statistics (Gram error, singular values) are computed for the
    FIRST bottleneck only -- they are identical by construction across layers,
    and a 4096x1024 SVD per layer would cost more than the initialization.
    """
    from .bottleneck.module import effective_backward_support

    stats: Dict = {"bottleneck_init": bn_init, "decoder_scale_mode": scale_mode}
    k_eff = g_d = None
    for mod in mods:
        k_eff = effective_backward_support(mod.gate)
        g_d = 1.0 if scale_mode == "none" else math.sqrt(d_model / k_eff)
        mod.decoder_scale = g_d
    stats.update({"k_eff": k_eff, "g_D": g_d, "bottlenecks": len(mods)})

    first = mods[0]
    stats["d_bottleneck"] = int(first.n_features)
    stats["k"], stats["j"] = int(first.gate.k), int(first.gate.j)
    if bn_init == "orthogonal":
        stats["encoder"] = _frame_stats(first.in_proj.weight, d_model)
        stats["decoder"] = _frame_stats(first.out_proj.weight, d_model)
        if g_d != 1.0:
            eff = _frame_stats(first.out_proj.weight * g_d, d_model)
            # the effective decoder's Gram targets (d_b/K_eff) I, not (d_b/d) I
            eff["gram_rel_err"] = _gram_rel_err(
                first.out_proj.weight * g_d, d_model,
                stats["d_bottleneck"] / k_eff)
            stats["decoder_effective"] = eff
    return stats


@torch.no_grad()
def _gram_rel_err(W: torch.Tensor, d_model: int, diag: float) -> float:
    """``||W W^T - diag*I||_F / ||diag*I||_F`` for a wide matrix, in float64."""
    Wd = W.detach().double()
    gram = Wd @ Wd.t()
    target = diag * torch.eye(d_model, dtype=gram.dtype, device=gram.device)
    return float((gram - target).norm() / target.norm())


@torch.no_grad()
def md_spread_gain_(p: torch.Tensor, scale: float, gain_mode: str = "row_col",
                    where: str = "split") -> Tuple[float, float]:
    """Put a scale into a matrix's MD **gains**, not into its sphere radius.

    Multiplies the fused weight by ``scale`` and places that factor in the
    matrix's gain vectors, by tagging the parameter with the initial gains the
    optimizer should start from.  ``DecoupledAdamW`` then sets
    ``softplus(raw)`` there and keeps ``c_F`` at the *direction's* norm, so
    ``W = diag(g_row) W_hat diag(g_col)`` still holds exactly at step 0.

    ``where`` decides the placement:

    ``"split"``  equally across the gain vectors the matrix has (both under
                 ``row_col``, the single one under ``up_down``)
    ``"row"``    all of it in ``g_row``   (one scalar per output row)
    ``"col"``    all of it in ``g_col``   (one scalar per input column)

    The fused weight is identical either way, so nothing about the forward or
    backward at step 0 depends on the placement; what differs is which
    parameters carry the factor, and therefore how training can move it.

    Scaling the fused weight on its own would put the factor into ``c_F``
    instead, where the sphere freezes it for the whole run.

    Returns ``(g_row, g_col)``.  Call it after ``md_init_``, before the
    optimizer is built.
    """
    row, col = _wants_gains(p.shape, gain_mode)
    if where not in ("split", "row", "col"):
        raise ValueError(f"unknown gain placement: {where!r} (split | row | col)")
    if where == "row" and not row:
        raise ValueError(f"a {tuple(p.shape)} matrix has no row gain under "
                         f"decouple_gains={gain_mode!r}")
    if where == "col" and not col:
        raise ValueError(f"a {tuple(p.shape)} matrix has no column gain under "
                         f"decouple_gains={gain_mode!r}")
    scale = float(scale)
    if where == "split":
        g = scale ** (1.0 / (int(row) + int(col)))
        gains = (g if row else 1.0, g if col else 1.0)
    else:
        gains = (scale if where == "row" else 1.0, scale if where == "col" else 1.0)
    p.mul_(scale)
    p._md_gain0 = gains  # read once, in DecoupledAdamW._state_for
    return gains


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
            # (1, 1) unless md_spread_gain_ asked for different starting gains
            g_row, g_col = getattr(p, "_md_gain0", (1.0, 1.0))
            raw = (lambda g: RAW_GAIN_ONE if g == 1.0 else math.log(math.expm1(g)))
            if row:
                state["raw_grow"] = torch.full((p.shape[0],), raw(g_row),
                                               device=p.device, dtype=p.dtype)
                state["grow_m"] = torch.zeros_like(state["raw_grow"])
                state["grow_v"] = torch.zeros_like(state["raw_grow"])
            if col:
                state["raw_gcol"] = torch.full((p.shape[1],), raw(g_col),
                                               device=p.device, dtype=p.dtype)
                state["gcol_m"] = torch.zeros_like(state["raw_gcol"])
                state["gcol_v"] = torch.zeros_like(state["raw_gcol"])
            # The sphere radius is the DIRECTION's initialization norm, captured
            # at first sight and kept in the state so resume preserves it.  With
            # gains at 1 that is just ||W||; a spread gain divides back out.
            c_f = p.detach().float().norm()
            if (g_row, g_col) != (1.0, 1.0):
                c_f = c_f / ((g_row if row else 1.0) * (g_col if col else 1.0))
            state["c_f"] = c_f.clone()
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


def build_decoupled_optimizer(model, train_cfg, gain_mode: str = "row_col"):
    """Group the model's parameters for :class:`DecoupledAdamW`.

    Embeddings (and the untied head) are the unit-row kind; every other dim>=2
    weight is a decoupled matrix; the rest (RMSNorm gains, biases) are plain
    Adam.  Weight decay is deliberately absent everywhere -- see the module
    docstring.
    """
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
