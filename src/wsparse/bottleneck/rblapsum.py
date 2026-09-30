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
                perm_rho=0.0, inactive_only=False):  # type: ignore[override]
        ctx.save_for_backward(value_c, active_c, score_c, sign_c, b, cap_active)
        ctx.t = float(t)
        ctx.mode_id = int(mode_id)
        ctx.k = int(k)
        ctx.sink = sink
        ctx.supp_scale = float(supp_scale)
        ctx.perm_rho = float(perm_rho)
        ctx.inactive_only = bool(inactive_only)
        return value_c * active_c

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        value_c, active_c, score_c, sign_c, b, cap_active = ctx.saved_tensors
        t, mode_id, k = ctx.t, ctx.mode_id, ctx.k

        # ordinary hard-forward path: dL/dz_i += upstream_i * m_i, unmodified
        grad_value = grad_y * active_c

        # the surrogate signal dL/dp_i = upstream_i * z_i.  The permutation
        # ablation scrambles a rho fraction of it WITHIN each row's pool,
        # BEFORE the kernel weighting: each position keeps its own kappa (its
        # distance to the boundary), so the surrogate's scale profile, kernel
        # locality and zero-sum structure are preserved and only the
        # assignment of task signal to neuron is destroyed.  The hard task
        # path above is exact and is never permuted.  rho=0 is a no-op on the
        # exact code path of the original backward.
        g_p = grad_y * value_c
        if ctx.perm_rho > 0.0:
            g_p = permute_fraction(g_p, ctx.perm_rho)

        # raw support gradient in score-margin space:  a = dL/dp * kappa
        kappa = laplace_pdf((score_c - b) / t) / t
        if ctx.inactive_only:
            # rblapsum_surrogate_scope="inactive": the support term only for the
            # J inactive members; active ones keep the exact hard gradient, and
            # every correction below is taken over the inactive members alone
            # (the (K+1)-st position is inactive by definition)
            kappa = kappa * (1 - active_c)
        a = g_p * kappa
        a_raw = a

        if mode_id == 1:  # project: remove the common-mode direction where cap binds
            if ctx.inactive_only:
                inact = 1 - active_c
                mean_in = a.sum(-1, keepdim=True) / inact.sum(-1, keepdim=True).clamp_min(1)
                a_proj = a - inact * mean_in
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

        g_s = a if ctx.supp_scale == 1.0 else a * ctx.supp_scale
        grad_value = grad_value + sign_c * g_s

        sink = ctx.sink
        if sink is not None:
            with torch.no_grad():
                gs = g_s.reshape(-1, g_s.shape[-1])
                nrm = gs.norm(dim=-1)
                sink["rb_support_grad_norm"] = nrm.mean().detach()
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
        return (grad_value, None, None, None, None, None, None, None, None, None,
                None, None, None)


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
                  inactive_only=False):
    """Apply the rank-boundary support gate; see :class:`_RBLapSumGate`.

    ``value_c`` (signed z at the sorted Top(K+J) candidates) carries gradient;
    all other tensors are detached inputs.  ``mode`` is one of :data:`GRAD_MODES`.
    ``supp_scale`` multiplies the surrogate support gradient in the backward
    (1.0 = normal; 0.0 = hard task path only) -- an experiment knob used by
    analysis/scale_dynamics.py for gradient-decomposition counterfactuals.
    ``inactive_only`` restricts the support term to the inactive pool members
    (rblapsum_surrogate_scope="inactive").
    """
    return _RBLapSumGate.apply(
        value_c, active_c, score_c.detach(), sign_c.detach(), b.detach(),
        float(t), _MODE_ID[mode], int(k), cap_active.detach(), sink,
        float(supp_scale), float(perm_rho), bool(inactive_only),
    )
