"""Rank-Boundary LapSum: hard TopK forward, local-kernel support gradient with
the boundary set by the hard rank (not by a soft mass constraint).

Keeps LapSum's Top(K+J) candidate set, its exponential local kernel, and its
hard forward, but drops the ``sum_i p_i = K`` mass constraint.  The boundary is

    b = max(b0, s_(K+1))            b0 = a fixed activation floor,

so the forward is an *upper* cap of K active features (fewer if fewer than K
scores exceed b0), never exactly K.  Selection score ``s`` is ``z`` for TopK
and ``|z|`` for AbsTopK, as elsewhere.

The support gradient is formulated in score-margin space.  With
``upstream_i = dL/dy_i`` and ``y_i = z_i m_i`` so ``dL/dm_i = upstream_i z_i``,
the raw per-candidate surrogate is

    a_i = upstream_i * z_i * kappa_T(s_i - b),     kappa_T(d) = e^{-|d|/T} / (2T)

(the LapSum ``laplace_pdf`` kernel, T a *fixed* constant -- deliberately not
score-scaled).  Three boundary-gradient modes differ ONLY in what they do to
``a`` before mapping back to z-space (``g_z = g_s`` for TopK,
``sign(z) g_s`` for AbsTopK):

    detach        g_s = a                     independent local gates
    project       g_s = a - mean(a)           remove the common-mode direction
                                              (only where the rank cap binds)
    through_rank  g_s = a; g_s[r] -= sum(a)   differentiate through the value of
                                              the (K+1)-st score, r its position
                                              (only where the rank cap binds)
    through_rank_kappa  g_s = a - q sum(a),   the same zero-sum correction
                  q = kappa/sum(kappa)        distributed kappa-weighted --
                                              LapSum's rank-one Jacobian form at
                                              the rank boundary (cap-active only)

``project`` and ``through_rank`` both give ``sum_i g_s_i = 0`` when the cap
binds, so a collective score shift produces no support gradient -- but neither
is LapSum's mass-preserving ``D_kappa - kappa kappa^T / sum(kappa)``.  When the
floor wins (``b = b0`` constant) both fall back to ``g_s = a``: a collective
lift off a fixed floor is a real change and must not be projected away.

Everything is O(K+J); no q x q matrix is ever formed.
"""

from __future__ import annotations

from typing import Optional

import torch

from .lapsum import laplace_cdf, laplace_pdf

GRAD_MODES = ("detach", "project", "through_rank", "through_rank_kappa")


def permute_fraction(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Randomly permute a ``rho`` fraction of each row's values, per row.

    For every row (token/block) independently: sample ``round(rho * q)``
    positions uniformly without replacement, and shuffle the VALUES at those
    positions by a uniform random permutation of the subset (fixed points
    allowed, as in any uniform permutation).  The multiset of values per row
    is preserved exactly; only the assignment moves.  Fresh randomness every
    call, torch's device RNG.  ``rho=0`` and subsets smaller than 2 return
    the input unchanged.
    """
    q = x.shape[-1]
    n_sel = int(round(rho * q))
    if n_sel < 2:
        return x
    rows = x.reshape(-1, q)
    sel = torch.rand_like(rows).argsort(-1)[:, :n_sel]        # random subset
    shuf = torch.rand(rows.shape[0], n_sel, device=x.device).argsort(-1)
    src = sel.gather(1, shuf)                                  # permuted subset
    out = rows.clone()
    out.scatter_(1, sel, rows.gather(1, src))
    return out.reshape(x.shape)
_MODE_ID = {"detach": 0, "project": 1, "through_rank": 2,
            "through_rank_kappa": 3}
_MEMBERS_ID = {"pool": 0, "inactive": 1, "active": 2}
# rblapsum_surrogate_scope -> which pool members get the support term; the
# "update*" scopes are also routed into a code-residual block's update
# (TransformerLM._code_residual_stack)
SURROGATE_SCOPES = {"pool": "pool", "inactive": "inactive", "update": "pool",
                    "update_inactive": "inactive", "update_active": "active",
                    "first_order": "pool", "first_order_inactive": "inactive"}
# The carry-aware scopes (code_residual only): the gate's output is split into
# the copy the next gate carries and the copy the decoder reads
# (:class:`_CarryReadSplit`); the carry's backward is the hard mask, and the
# support term is driven by the READ gradients only, never by a gradient that
# arrived along the carry from another gate's support term.  The suffix names
# how far downstream a flip at this gate is assumed to be felt:
#   local       the next block's read only (the stream-carried surrogate's
#               logic: D^T of the gradient at this bottleneck's own decoder)
#   hard        plus the downstream reads the coordinate reaches along the hard
#               carry (the actual masks: for an inactive candidate, usually none)
#   persistent  plus every downstream read, as if the flipped coordinate stayed
#               in the code to the end
#   mixed       hard for the active members (eviction removes what the carry
#               would have delivered), persistent for the inactive ones (an
#               entering feature is carried from here on)
# ``_inactive`` restricts the term to the J inactive members.
CARRY_DRIVERS = {"carry_local": ("local", "local"),
                 "carry_hard": ("hard", "hard"),
                 "carry_persistent": ("persistent", "persistent"),
                 "carry_mixed": ("hard", "persistent")}
for _name in list(CARRY_DRIVERS):
    SURROGATE_SCOPES[_name] = "pool"
    SURROGATE_SCOPES[_name + "_inactive"] = "inactive"
    CARRY_DRIVERS[_name + "_inactive"] = CARRY_DRIVERS[_name]
_DRIVER_ID = {"local": 0, "hard": 1, "persistent": 2}


def carry_scope(scope: str) -> bool:
    return scope in CARRY_DRIVERS


class FirstOrderState:
    """Which backward pass is running, for rblapsum_surrogate_scope="first_order".

    The first-order estimator needs, at every gate, the gradient that reached
    it along hard paths only.  :func:`first_order_backward` runs a backward
    with ``phase = "hard"`` (every first-order gate passes back only ``m * g``
    and keeps that ``g``), then the ordinary one, in which each such gate adds
    its support term computed from the kept hard-path gradient instead of from
    the total upstream.  One process trains one model, so the state is global.
    """

    def __init__(self) -> None:
        self.phase = "total"


FIRST_ORDER = FirstOrderState()


class first_order_hard_pass:
    """Context manager: the backward inside it is the hard pass."""

    def __enter__(self):
        FIRST_ORDER.phase = "hard"
        return self

    def __exit__(self, *exc):
        FIRST_ORDER.phase = "total"
        return False


def first_order_backward(loss: torch.Tensor, anchor: torch.Tensor) -> None:
    """``loss.backward()`` with the first-order support terms.

    ``anchor`` is a leaf below every gate (the token embedding): the hard pass
    is ``torch.autograd.grad`` to it, so no parameter's ``.grad`` is touched,
    and only the second pass accumulates.
    """
    with first_order_hard_pass():
        torch.autograd.grad(loss, [anchor], retain_graph=True, allow_unused=True)
    loss.backward()


def center_over_tokens(g_s: torch.Tensor, cand_idx: torch.Tensor,
                       n_features: int) -> torch.Tensor:
    """Remove, per feature, the token mean of the support term.

    ``g_s`` (score space) and ``cand_idx`` are ``(..., K+J)``, one row per
    token.  For every feature the mean of its support term over the rows whose
    pool contains it is subtracted on those rows, so the term sums to zero
    over the tokens for every feature (rblapsum_center_tokens).  Complements
    the per-token zero-sum of the boundary correction: that one forbids
    moving all of a token's scores together, this one moving one feature's
    score for all tokens together.
    """
    q = g_s.shape[-1]
    gs = g_s.reshape(-1, q)
    idx = cand_idx.reshape(-1, q)
    flat = idx.reshape(-1)
    total = torch.zeros(n_features, dtype=gs.dtype, device=gs.device).index_add_(
        0, flat, gs.reshape(-1))
    count = torch.zeros(n_features, dtype=gs.dtype, device=gs.device).index_add_(
        0, flat, torch.ones_like(gs).reshape(-1))
    mean = total / count.clamp_min(1)
    return (gs - mean[idx]).reshape(g_s.shape)


def relative_kernel_width(t: float, b: torch.Tensor) -> torch.Tensor:
    """The scale-free kernel width ``t * b`` (rblapsum_relative_temperature).

    ``b`` is the per-row rank boundary; the floor keeps the kernel finite on a
    row whose boundary is zero (fewer than K+1 nonzero scores).
    """
    return (float(t) * b).clamp_min(1e-6)


def kernel_width(mode: str, t: float, score_c: torch.Tensor, b: torch.Tensor, k: int,
                 j: int):
    """Per-row kernel width for rblapsum_kernel_width.

    ``fixed``: the float ``t``.  ``relative_b``: ``t * b``.  ``relative_span``:
    ``t * (s_(K+1) - s_(K+J))`` over the sorted pool ``score_c`` -- the score
    interval the J candidates occupy, floored so a degenerate pool (ties, or
    fewer than K+J nonzero scores) keeps a finite kernel.
    """
    if mode == "fixed":
        return float(t)
    if mode == "relative_b":
        return relative_kernel_width(t, b)
    if mode == "relative_span":
        span = score_c[..., k:k + 1] - score_c[..., k + j - 1:k + j]
        return (float(t) * span).clamp_min(1e-6)
    raise ValueError(f"unknown rblapsum_kernel_width: {mode!r}")


def strength_scale(s: float, t, b: torch.Tensor) -> torch.Tensor:
    """Per-row support scale ``2 s T / b`` (rblapsum_support_strength).

    Makes ``gamma * |u| kappa = s`` for a member at the boundary, where
    ``|u| = b`` and ``kappa = 1 / 2T``.
    """
    return 2.0 * float(s) * t / b.clamp_min(1e-6)



def support_term(g_up, value_c, active_c, score_c, sign_c, b, t, mode_id, k,
             cap_active, supp_scale, perm_rho, members, cand_idx, n_features,
             sink):
    """The RBLapSum support term in score space over the sorted pool.

    ``g_up`` is the upstream the term is driven by (``dL/dy`` over the pool, or
    the hard-path gradient under the first-order and carry scopes); the
    result ``g_s`` is added to ``dL/dz`` as ``sign_c * g_s``.  Shared by the
    gate Functions so that every scope applies exactly the same term.
    """
    # the surrogate signal dL/dp_i = upstream_i * z_i.  The permutation
    # ablation scrambles a rho fraction of it WITHIN each row's pool,
    # BEFORE the kernel weighting: each position keeps its own kappa (its
    # distance to the boundary), so the surrogate's scale profile, kernel
    # locality and zero-sum structure are preserved and only the
    # assignment of task signal to neuron is destroyed.  The hard task
    # path above is exact and is never permuted.  rho=0 is a no-op on the
    # exact code path of the original backward.
    g_p = g_up * value_c
    if perm_rho > 0.0:
        g_p = permute_fraction(g_p, perm_rho)

    # raw support gradient in score-margin space:  a = dL/dp * kappa
    kappa = laplace_pdf((score_c - b) / t) / t
    member = None
    if members:
        # rblapsum_surrogate_scope restricted to one side of the boundary:
        # 1 = the J inactive members (active ones keep the exact hard
        # gradient), 2 = the K active members (inactive ones get none);
        # every correction below is taken over those members alone
        member = (1 - active_c) if members == 1 else active_c
        kappa = kappa * member
    a = g_p * kappa
    a_raw = a

    if mode_id == 1:  # project: remove the common-mode direction where cap binds
        if member is not None:
            mean_m = a.sum(-1, keepdim=True) / member.sum(-1, keepdim=True).clamp_min(1)
            a_proj = a - member * mean_m
        else:
            a_proj = a - a.mean(-1, keepdim=True)
        a = torch.where(cap_active, a_proj, a)
    elif mode_id == 2:  # through_rank: -sum(a) onto the (K+1)-st position
        total = a.sum(-1, keepdim=True)
        corr = torch.zeros_like(a)
        corr[..., k:k + 1] = torch.where(cap_active, total, torch.zeros_like(total))
        a = a - corr
    elif mode_id == 3:  # through_rank_kappa: the same zero-sum correction,
        # distributed kappa-weighted across the pool instead of as a point
        # mass on the boundary feature.  This is exactly LapSum's rank-one
        # Jacobian structure (g = a - q * sum(a), q = kappa / sum kappa)
        # applied at the rank boundary: each member's common mode is removed
        # IN PROPORTION TO ITS OWN kernel weight, so no feature is left
        # uncompensated and none becomes a sink.  Cap-active rows only.
        total = a.sum(-1, keepdim=True)
        q = kappa / kappa.sum(-1, keepdim=True).clamp_min(
            torch.finfo(kappa.dtype).tiny)
        a = torch.where(cap_active, a - q.to(a.dtype) * total, a)

    if isinstance(supp_scale, torch.Tensor):
        g_s = a * supp_scale.to(a.dtype)       # per-row scale (rblapsum_support_strength)
    else:
        g_s = a if supp_scale == 1.0 else a * supp_scale
    if cand_idx is not None:
        g_s = center_over_tokens(g_s, cand_idx, n_features)

    if sink is not None:
        with torch.no_grad():
            gs = g_s.reshape(-1, g_s.shape[-1])
            nrm = gs.norm(dim=-1)
            sink["rb_support_grad_norm"] = nrm.mean().detach()
            if isinstance(supp_scale, torch.Tensor):
                sink["rb_support_scale_eff"] = supp_scale.float().mean().detach()
            if isinstance(t, torch.Tensor):
                sink["rb_temperature_eff"] = t.float().mean().detach()
            eps = torch.finfo(gs.dtype).eps
            sink["rb_common_mode"] = (
                gs.sum(-1).abs() / (nrm + eps)
            ).mean().detach()
            ar = a_raw.reshape(-1, a_raw.shape[-1])
            sink["rb_common_mode_raw"] = (
                ar.sum(-1).abs() / (ar.norm(dim=-1) + eps)
            ).mean().detach()
            # mean |kick| inside the kernel window -- the temperature
            # servo's raw pressure signal (read at the next forward)
            win = (score_c - b).abs() < t
            sink["rb_kick_win"] = (
                g_s.abs()[win].mean().detach() if bool(win.any())
                else torch.zeros((), device=g_s.device))
            if mode_id == 2:
                # is the boundary feature becoming a gradient sink?
                bmag = gs[:, k].abs().mean()
                omag = gs.abs().mean()
                sink["rb_boundary_grad_ratio"] = (bmag / (omag + eps)).detach()
    return g_s


class _RBLapSumGate(torch.autograd.Function):
    """``y_c = value_c * active_c`` with the rank-boundary support gradient.

    Only ``value_c`` (the candidate z-values) carries gradient; ``score_c``,
    ``sign_c``, ``b`` and ``active_c`` enter detached, so autograd never flows
    through the score = |z| map or the boundary except where this backward
    routes it explicitly.  The candidate array is sorted descending, so the
    (K+1)-st feature -- the rank boundary -- is at fixed position ``k``.
    """

    @staticmethod
    def forward(ctx, value_c, active_c, score_c, sign_c, b, t, mode_id, k,
                cap_active, sink, supp_scale=1.0,
                perm_rho=0.0, members=0,
                relative_t=False, cand_idx=None,
                n_features=0, first_order=False):  # type: ignore[override]
        ctx.save_for_backward(value_c, active_c, score_c, sign_c, b, cap_active,
                              cand_idx)
        ctx.n_features = int(n_features)
        # t and supp_scale may be per-row tensors (rblapsum_kernel_width,
        # rblapsum_support_strength); they are detached inputs either way
        ctx.t = t if isinstance(t, torch.Tensor) else float(t)
        ctx.mode_id = int(mode_id)
        ctx.k = int(k)
        ctx.sink = sink
        ctx.supp_scale = supp_scale if isinstance(supp_scale, torch.Tensor) else float(supp_scale)
        ctx.perm_rho = float(perm_rho)
        ctx.members = int(members)
        ctx.relative_t = bool(relative_t)
        # rblapsum_surrogate_scope="first_order": the hard-path gradient kept by
        # the hard pass, used for the support term in the total pass
        ctx.first_order = bool(first_order)
        ctx.hard_upstream = None
        return value_c * active_c

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        value_c, active_c, score_c, sign_c, b, cap_active, cand_idx = ctx.saved_tensors
        t, mode_id, k = ctx.t, ctx.mode_id, ctx.k
        if ctx.relative_t:
            # scale-free kernel: width t * b per row (rblapsum_relative_temperature)
            t = relative_kernel_width(t, b)

        # ordinary hard-forward path: dL/dz_i += upstream_i * m_i, unmodified
        grad_value = grad_y * active_c
        g_up = grad_y
        if ctx.first_order:
            if FIRST_ORDER.phase == "hard":
                # the hard pass: keep the hard-path gradient, pass back m * g
                ctx.hard_upstream = grad_y.detach()
                return (grad_value,) + (None,) * 16
            if ctx.hard_upstream is not None:
                # the total pass: one support term per gate, from the hard path
                # (a jump at this gate times exact Jacobians elsewhere), never
                # from another gate's support term
                g_up, ctx.hard_upstream = ctx.hard_upstream, None

        g_s = support_term(g_up, value_c, active_c, score_c, sign_c, b, t, mode_id, k,
                           cap_active, ctx.supp_scale, ctx.perm_rho, ctx.members,
                           cand_idx, ctx.n_features, ctx.sink)
        grad_value = grad_value + sign_c * g_s
        return (grad_value, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None)


class _RBLapSumSFGate(torch.autograd.Function):
    """Soft forward ``y_c = value_c * p_c`` with ``p = F((s - b)/T)``.

    The soft-forward (sf) sibling of :class:`_RBLapSumGate`: the SAME
    Top(K+J) pool, rank boundary ``b = max(b0, s_(K+1))`` and exponential
    kernel, but the probabilities are IN the forward -- every pool member
    outputs ``z_i * p_i`` (features outside the pool output exactly 0) -- and
    the backward is the gradient of that forward, no surrogate discrepancy.

    Because ``F' = kappa``, the direct term of the score gradient is the
    familiar ``a_i = upstream_i * z_i * kappa_i``: exactly the ``a`` of the
    hard gate.  The modes differ only in how the boundary's own derivative is
    routed, written uniformly as

        g_s = a - supp_scale * dist * sum(a)

    with ``dist`` = 0 (``detach``), uniform (``project``), a point mass on the
    (K+1)-st position (``through_rank``), or ``kappa/sum(kappa)``
    (``through_rank_kappa``); the correction applies only where the rank cap
    binds (a floor boundary is a constant -- no derivative to route).
    ``through_rank`` at ``supp_scale=1`` is the EXACT autograd gradient of the
    forward, since ``b`` literally is the (K+1)-st score; the kappa mode
    deposits the same total kappa-weighted, the same modification as in the
    hard gate.  Unlike the hard gate -- where ``supp_scale`` multiplies the
    whole surrogate ``g_s`` -- here it multiplies ONLY the boundary term:
    the direct path is a true forward gradient, and 0.0 reproduces
    ``detach`` rather than a hard-TopK backward.

    The value path is the true one by default: ``dL/dz_i += upstream_i * p_i``,
    so inactive pool members receive value gradient in proportion to their
    probability.  ``value_grad_id=1`` (``"support"``) masks that path to the
    hard support: the J inactive candidates then receive ONLY the score-path
    term -- their values are no longer trained, while their ranking still is.
    The forward is unchanged, so this deliberately reintroduces a
    forward/backward discrepancy on the tail; active features keep the sf
    value gradient ``u * p`` (not the hard gate's ``u * 1``).
    """

    @staticmethod
    def forward(ctx, value_c, p_c, active_c, score_c, sign_c, b, t, mode_id,
                k, cap_active, sink, supp_scale=1.0,
                value_grad_id=0):  # type: ignore[override]
        ctx.save_for_backward(value_c, p_c, active_c, score_c, sign_c, b,
                              cap_active)
        ctx.t = float(t)
        ctx.mode_id = int(mode_id)
        ctx.k = int(k)
        ctx.sink = sink
        ctx.supp_scale = float(supp_scale)
        ctx.value_grad_id = int(value_grad_id)
        return value_c * p_c

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        (value_c, p_c, active_c, score_c, sign_c, b,
         cap_active) = ctx.saved_tensors
        t, mode_id, k = ctx.t, ctx.mode_id, ctx.k

        # value path of y = z * p:  dL/dz_i += upstream_i * p_i, masked to the
        # hard support under value_grad_id=1 ("support")
        grad_value = grad_y * p_c
        if ctx.value_grad_id == 1:
            grad_value = grad_value * active_c

        # direct score term (dp/ds = kappa):  a = upstream * z * kappa
        kappa = laplace_pdf((score_c - b) / t) / t
        a = grad_y * value_c * kappa

        g_s = a
        if mode_id and ctx.supp_scale != 0.0:
            total = a.sum(-1, keepdim=True) * ctx.supp_scale
            if mode_id == 1:  # project: uniform distribution
                corr = total / a.shape[-1]
            elif mode_id == 2:  # through_rank: point mass on the (K+1)-st position
                corr = torch.zeros_like(a)
                corr[..., k:k + 1] = total
            else:  # through_rank_kappa
                q = kappa / kappa.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(kappa.dtype).tiny)
                corr = q.to(a.dtype) * total
            g_s = torch.where(cap_active, a - corr, a)

        grad_value = grad_value + sign_c * g_s

        sink = ctx.sink
        if sink is not None:
            with torch.no_grad():
                gs = g_s.reshape(-1, g_s.shape[-1])
                nrm = gs.norm(dim=-1)
                eps = torch.finfo(gs.dtype).eps
                sink["rb_support_grad_norm"] = nrm.mean().detach()
                sink["rb_common_mode"] = (
                    gs.sum(-1).abs() / (nrm + eps)
                ).mean().detach()
                ar = a.reshape(-1, a.shape[-1])
                sink["rb_common_mode_raw"] = (
                    ar.sum(-1).abs() / (ar.norm(dim=-1) + eps)
                ).mean().detach()
                win = (score_c - b).abs() < t
                sink["rb_kick_win"] = (
                    g_s.abs()[win].mean().detach() if bool(win.any())
                    else torch.zeros((), device=g_s.device))
        return (grad_value, None, None, None, None, None, None, None, None,
                None, None, None, None)


VALUE_GRAD_MODES = ("pool", "support")
_VALUE_GRAD_ID = {"pool": 0, "support": 1}


def rblapsum_sf_gate(value_c, p_c, active_c, score_c, sign_c, b, t, mode, k,
                     cap_active, sink=None, supp_scale=1.0,
                     value_grad="pool"):
    """Apply the soft-forward gate; see :class:`_RBLapSumSFGate`.

    ``value_c`` carries gradient; ``p_c`` and the rest enter detached (the
    backward routes the score/boundary derivatives explicitly).
    ``supp_scale`` multiplies only the boundary term of the score gradient.
    ``value_grad`` is "pool" (every candidate's value trains, the gradient of
    the forward) or "support" (the J inactive candidates receive only the
    score-path term).
    """
    return _RBLapSumSFGate.apply(
        value_c, p_c.detach(), active_c.detach(), score_c.detach(),
        sign_c.detach(), b.detach(), float(t), _MODE_ID[mode], int(k),
        cap_active.detach(), sink, float(supp_scale),
        _VALUE_GRAD_ID[value_grad],
    )


def rblapsum_gate(value_c, active_c, score_c, sign_c, b, t, mode, k,
                  cap_active, sink=None, supp_scale=1.0, perm_rho=0.0,
                  members="pool", relative_t=False, cand_idx=None, n_features=0,
                  first_order=False):
    """Apply the rank-boundary support gate; see :class:`_RBLapSumGate`.

    ``value_c`` (signed z at the sorted Top(K+J) candidates) carries gradient;
    all other tensors are detached inputs.  ``mode`` is one of :data:`GRAD_MODES`.
    ``supp_scale`` multiplies the surrogate support gradient in the backward
    (1.0 = normal; 0.0 = hard task path only) -- an experiment knob used by
    analysis/scale_dynamics.py for gradient-decomposition counterfactuals.
    ``members`` restricts the support term to the "inactive" or the "active"
    pool members ("pool": all of them; rblapsum_surrogate_scope).  ``relative_t`` reads ``t`` as a
    width relative to the boundary: the kernel is ``kappa_{t b}`` per row
    (rblapsum_relative_temperature).  With ``cand_idx`` (the candidates'
    feature indices) and ``n_features`` the support term is centered over the
    tokens per feature (rblapsum_center_tokens, :func:`center_over_tokens`).
    """
    return _RBLapSumGate.apply(
        value_c, active_c, score_c.detach(), sign_c.detach(), b.detach(),
        t.detach() if isinstance(t, torch.Tensor) else float(t),
        _MODE_ID[mode], int(k), cap_active.detach(), sink,
        supp_scale.detach() if isinstance(supp_scale, torch.Tensor) else float(supp_scale),
        float(perm_rho), _MEMBERS_ID[members], bool(relative_t),
        None if cand_idx is None else cand_idx.detach(), int(n_features),
        bool(first_order),
    )


# ---- carry-aware scopes (code_residual) ------------------------------------- #

class CarryChain:
    """Per-forward backward state of a code-residual stack under a carry scope.

    ``state[l]`` is written in two steps during the backward: the split of
    gate ``l``'s output stores ``"r"``, the gradient the decoder read of that
    output sends back (over all N coordinates); gate ``l``'s own backward then
    replaces it by ``"H"`` and ``"R"``, the two per-coordinate sums it hands
    to gate ``l-1``:

        H_l = m_l * (r_{l+1} + H_{l+1})   reads reached along the hard carry
        R_l = r_{l+1} + R_{l+1}           every downstream read

    (``m_l`` this gate's mask, ``H = R = 0`` above the last gate).  The model
    creates one chain per forward (``TransformerLM._code_residual_stack``), so
    micro-batches never share state; gate ``l+1``'s backward precedes gate
    ``l``'s because ``u_{l+1}`` depends on ``c_{l+1}``.
    """

    def __init__(self) -> None:
        self.state: dict = {}


class _CarryReadSplit(torch.autograd.Function):
    """``y -> (y_carry, y_read)``: two copies of the gate's output.

    The next gate carries the first, the decoder reads the second.  The
    backward returns their sum to the gate (the total gradient, as before) and
    keeps the read gradient in the chain so that the gate's backward can tell
    the two apart.
    """

    @staticmethod
    def forward(ctx, y, chain, index):  # type: ignore[override]
        ctx.chain = chain
        ctx.index = int(index)
        return y.clone(), y.clone()

    @staticmethod
    def backward(ctx, g_carry, g_read):  # type: ignore[override]
        ctx.chain.state.setdefault(ctx.index, {})["r"] = g_read
        return g_carry + g_read, None, None


class _RBLapSumCarryGate(torch.autograd.Function):
    """:class:`_RBLapSumGate` for the carry scopes.

    The forward is the hard gate.  In the backward the hard mask passes the
    TOTAL upstream (carry + read) back to the gate's input, as the hard model
    does, and the support term is driven by the read gradients only::

        g_up_i = r_i + H_i   (active members, "hard"; eviction loses what the
                              hard carry would have delivered downstream)
        g_up_i = r_i + R_i   (inactive members, "persistent"; an entering
                              feature is carried from here on)
        g_up_i = r_i         ("local": the next block's read only)

    with ``r`` this gate's own read gradient and ``H``, ``R`` the chain sums of
    the gate above (:class:`CarryChain`).  No gradient that arrived along the
    carry from another gate's support term ever enters a kernel, so the
    products of support terms along the identity carry -- the term that grows
    exponentially with depth under the ``pool`` scope -- do not exist; the
    support terms of the gates above still reach this gate's input, added,
    through the mask.
    """

    @staticmethod
    def forward(ctx, value_c, active_c, score_c, sign_c, b, t, mode_id, k,
                cap_active, sink, supp_scale, members, relative_t, cand_idx,
                n_features, chain, index, evict_id, entry_id,
                decay=1.0):  # type: ignore[override]
        ctx.save_for_backward(value_c, active_c, score_c, sign_c, b, cap_active,
                              cand_idx)
        # rblapsum_carry_decay: R_l = r_{l+1} + decay * R_{l+1}, the expected
        # number of downstream reads of an entering feature when it survives
        # each further gate with probability `decay` (1 = persistent)
        ctx.decay = float(decay)
        ctx.n_features = int(n_features)
        ctx.t = float(t)
        ctx.mode_id = int(mode_id)
        ctx.k = int(k)
        ctx.sink = sink
        ctx.supp_scale = float(supp_scale)
        ctx.members = int(members)
        ctx.relative_t = bool(relative_t)
        ctx.chain = chain
        ctx.index = int(index)
        ctx.evict_id = int(evict_id)
        ctx.entry_id = int(entry_id)
        return value_c * active_c

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        value_c, active_c, score_c, sign_c, b, cap_active, cand_idx = ctx.saved_tensors
        t, mode_id, k = ctx.t, ctx.mode_id, ctx.k
        if ctx.relative_t:
            t = relative_kernel_width(t, b)
        state = ctx.chain.state
        mine = state.get(ctx.index)
        if mine is None or "r" not in mine:
            raise RuntimeError(
                "carry scope: the gate's output was not split into carry and read "
                "copies (TransformerLM._code_residual_stack does this)")
        r_full = mine.pop("r")
        above = state.pop(ctx.index + 1, None)

        # the hard path on the total upstream, as in every other scope
        grad_value = grad_y * active_c

        r_pool = r_full.gather(-1, cand_idx)
        if above is None:
            h_full = r_full
            rr_full = r_full
            h_pool = rr_pool = r_pool
        else:
            h_full = r_full + above["H"]
            rr_full = (r_full + above["R"] if ctx.decay == 1.0
                       else r_full + ctx.decay * above["R"])
            h_pool = h_full.gather(-1, cand_idx)
            rr_pool = rr_full.gather(-1, cand_idx)

        def driver(did):
            return r_pool if did == 0 else (h_pool if did == 1 else rr_pool)

        g_up = (driver(ctx.evict_id) if ctx.evict_id == ctx.entry_id
                else torch.where(active_c > 0, driver(ctx.evict_id),
                                 driver(ctx.entry_id)))
        g_s = support_term(g_up, value_c, active_c, score_c, sign_c, b, t, mode_id, k,
                           cap_active, ctx.supp_scale, 0.0, ctx.members,
                           None, ctx.n_features, ctx.sink)
        grad_value = grad_value + sign_c * g_s

        # the chain for the gate below: H masked by THIS gate's support
        mask_full = torch.zeros_like(r_full).scatter(-1, cand_idx, active_c.to(r_full.dtype))
        state[ctx.index] = {"H": h_full * mask_full, "R": rr_full}
        return (grad_value,) + (None,) * 19


def rblapsum_carry_gate(value_c, active_c, score_c, sign_c, b, t, mode, k,
                        cap_active, sink, supp_scale, members, relative_t,
                        cand_idx, n_features, chain, index, scope, decay=1.0):
    """Apply the carry-scope gate; see :class:`_RBLapSumCarryGate`.

    ``decay`` is rblapsum_carry_decay, the per-gate factor on the persistent
    chain (1.0: every downstream read counts in full).
    """
    evict, entry = CARRY_DRIVERS[scope]
    return _RBLapSumCarryGate.apply(
        value_c, active_c, score_c.detach(), sign_c.detach(), b.detach(),
        float(t), _MODE_ID[mode], int(k), cap_active.detach(), sink,
        float(supp_scale), _MEMBERS_ID[members], bool(relative_t),
        cand_idx.detach(), int(n_features), chain, int(index),
        _DRIVER_ID[evict], _DRIVER_ID[entry], float(decay),
    )


def split_carry_read(y, chain, index):
    """``(y_carry, y_read)`` for a gate's output under a carry scope."""
    return _CarryReadSplit.apply(y, chain, index)
