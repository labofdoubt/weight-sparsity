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

from .lapsum import laplace_pdf

GRAD_MODES = ("detach", "project", "through_rank", "through_rank_kappa")
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
                cap_active, sink):  # type: ignore[override]
        ctx.save_for_backward(value_c, active_c, score_c, sign_c, b, cap_active)
        ctx.t = float(t)
        ctx.mode_id = int(mode_id)
        ctx.k = int(k)
        ctx.sink = sink
        return value_c * active_c

    @staticmethod
    def backward(ctx, grad_y):  # type: ignore[override]
        value_c, active_c, score_c, sign_c, b, cap_active = ctx.saved_tensors
        t, mode_id, k = ctx.t, ctx.mode_id, ctx.k

        # ordinary hard-forward path: dL/dz_i += upstream_i * m_i, unmodified
        grad_value = grad_y * active_c

        # raw support gradient in score-margin space:  a = upstream * z * kappa
        kappa = laplace_pdf((score_c - b) / t) / t
        a = grad_y * value_c * kappa
        a_raw = a

        if mode_id == 1:  # project: remove the common-mode direction where cap binds
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

        g_s = a
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
                if mode_id == 2:
                    # is the boundary feature becoming a gradient sink?
                    bmag = gs[:, k].abs().mean()
                    omag = gs.abs().mean()
                    sink["rb_boundary_grad_ratio"] = (bmag / (omag + eps)).detach()
        return (grad_value, None, None, None, None, None, None, None, None, None)


def rblapsum_gate(value_c, active_c, score_c, sign_c, b, t, mode, k,
                  cap_active, sink=None):
    """Apply the rank-boundary support gate; see :class:`_RBLapSumGate`.

    ``value_c`` (signed z at the sorted Top(K+J) candidates) carries gradient;
    all other tensors are detached inputs.  ``mode`` is one of :data:`GRAD_MODES`.
    """
    return _RBLapSumGate.apply(
        value_c, active_c, score_c.detach(), sign_c.detach(), b.detach(),
        float(t), _MODE_ID[mode], int(k), cap_active.detach(), sink,
    )
