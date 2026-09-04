"""Adaptive LapSum temperature: pick ``t`` from a target effective count.

``t`` is a gradient *bandwidth*, not a learned parameter.  The score scale drifts
across tokens, across layers and over training, so a fixed numeric ``t`` would
mean a different thing everywhere.  Instead each row solves for the ``t`` whose
boundary-exchange weight distribution has a prescribed effective size ``N_eff``.

**The adaptive temperature is a detached per-token bandwidth.**  The backward is
the exact LapSum VJP conditional on that fixed bandwidth; it is *not* the
derivative of the full ``t(r)`` solver.  Nothing here differentiates through a
Newton or bisection iteration, and the ``N_eff`` calibration equation
contributes no gradient at all.  The backward does still account for
``b = b(r; t)`` through ``sum_i p_i = K``, which is what the LapSum VJP encodes.
This is a deliberate experimental choice, not an implementation artefact.

Two definitions of "effective size", over a normalized weight vector ``q``:

    ESS       N_eff = 1 / sum_i q_i^2       (ignores a long tail of small weights)
    entropy   N_eff = exp(-sum_i q_i log q_i)  (counts the tail more)

and three ways to choose the weights that get calibrated:

``outside_only`` + ``score_softmax``
    ``q_i = softmax(r_i / t)`` over the ``J`` inactive candidates.  The cheap
    decoupled **approximation**: ``q`` does not involve the barrier at all, so
    ``t`` and ``b`` solve independently -- a scalar root-find over ``J`` scores,
    then one closed-form barrier solve, no barrier inside the loop.  It equals
    the normalized LapSum gradient weights only when every outside candidate
    lies below the barrier (``r_{K+1} < b``); at finite temperature the
    budget-preserving barrier can move above ``r_{K+1}`` and then it does not.

``outside_only`` + ``true_gradient``
    ``q_i = kappa_i / sum_{j > K} kappa_j``, the *actual* LapSum gradient
    weights restricted to the inactive candidates.  Exact, but ``kappa`` depends
    on ``b``, so this does not decouple.

``both_sides``
    the same actual gradient weights over all ``M = K + J`` candidates.

The last two share one solver: they differ only in which indices contribute to
the effective-count equation, passed as ``calibration``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .lapsum import laplace_cdf, lapsum_barrier_sorted

_TAU_CLIP = 1.0  # max |delta log t| per Newton step
_B_CLIP = 4.0  # max |delta b| per Newton step, in units of t

# Solver status codes, reported per row rather than raised: a target can be
# unattainable for some token rows and fine for others in the same batch.
STATUS_OK = 0
STATUS_BELOW_RANGE = 1  # target under the small-t limit of N_eff
STATUS_ABOVE_RANGE = 2  # target over the large-t limit of N_eff
STATUS_DEGENERATE = 3  # tied scores: N_eff does not respond to t at all

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_BELOW_RANGE: "target_below_attainable_range",
    STATUS_ABOVE_RANGE: "target_above_attainable_range",
    STATUS_DEGENERATE: "degenerate_scores",
}


def effective_count(q: torch.Tensor, metric: str) -> torch.Tensor:
    """``N_eff`` of a normalized weight vector."""
    if metric == "ess":
        return 1.0 / (q * q).sum(-1).clamp_min(torch.finfo(q.dtype).tiny)
    if metric == "entropy":
        return torch.exp(-torch.xlogy(q, q).sum(-1))
    raise ValueError(f"unknown effective_count_metric: {metric!r} (ess | entropy)")


def _count_and_coeff(logits: torch.Tensor, metric: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(N_eff, dN/dlogits)`` for ``q = softmax(logits)``."""
    q = torch.softmax(logits, dim=-1)
    if metric == "ess":
        s2 = (q * q).sum(-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        n = 1.0 / s2
        coeff = -2.0 * n * n * q * (q - s2)
        return n.squeeze(-1), coeff
    if metric == "entropy":
        n = torch.exp(-torch.xlogy(q, q).sum(-1, keepdim=True))
        mean_l = (q * logits).sum(-1, keepdim=True)
        coeff = n * q * (mean_l - logits)
        return n.squeeze(-1), coeff
    raise ValueError(f"unknown effective_count_metric: {metric!r} (ess | entropy)")


# --------------------------------------------------------------------------- #
# realized effective counts (diagnostics + reference checks)
# --------------------------------------------------------------------------- #


def score_softmax_count(inactive: torch.Tensor, t: torch.Tensor, metric: str) -> torch.Tensor:
    """``N_eff`` of the cheap ``softmax(r / t)`` weights over the inactive tail."""
    centred = inactive - inactive[..., :1]
    return effective_count(torch.softmax(centred / t.unsqueeze(-1), dim=-1), metric)


def gradient_weights(
    candidates: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    calibration: slice = slice(None),
) -> torch.Tensor:
    """``kappa_i / sum_j kappa_j`` over the calibration subset.

    The ``1/(2t)`` prefactor cancels, leaving ``softmax(-|z|)``.
    """
    z = (candidates - b.unsqueeze(-1)) / t.unsqueeze(-1)
    return torch.softmax(-z[..., calibration].abs(), dim=-1)


def gradient_count(
    candidates: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    metric: str,
    calibration: slice = slice(None),
) -> torch.Tensor:
    """``N_eff`` of the true LapSum gradient weights over the calibration subset."""
    return effective_count(gradient_weights(candidates, b, t, calibration), metric)


# --------------------------------------------------------------------------- #
# one-sided score_softmax: the cheap decoupled approximation
# --------------------------------------------------------------------------- #


def solve_score_softmax_temperature(
    inactive: torch.Tensor,
    n_eff_target: float,
    metric: str = "ess",
    tol: float = 1e-5,
    max_iters: int = 12,
    fallback_scale: Optional[torch.Tensor] = None,
    initial_log_range: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Solve ``N_eff(softmax(r/t)) = n_eff_target`` over the ``J`` inactive scores.

    The barrier does not appear, so this needs no barrier solve at all.  It is
    the cheap approximation, and it is also the initializer for the exact
    solvers below.

    Solved in ``tau = log t``, which guarantees ``t > 0`` and makes the
    multiplicative updates convenient.  Scale equivariance comes from elsewhere:
    the calibration equation depends only on score *differences* divided by
    ``t``, and the bracket below is built from the score span, so ``r -> c r``
    maps ``t -> c t`` exactly.
    """
    with torch.no_grad():
        r = inactive.float()
        j = r.shape[-1]
        centred = r - r[..., :1]  # <= 0, first entry 0
        span = (-centred[..., -1]).clamp_min(0.0)
        tiny = torch.finfo(r.dtype).tiny

        if fallback_scale is None:
            fallback_scale = torch.ones_like(span)
        degenerate = span <= tiny
        scale = torch.where(degenerate, fallback_scale.clamp_min(tiny), span)
        target = float(n_eff_target)

        def count(tau: torch.Tensor) -> torch.Tensor:
            return effective_count(
                torch.softmax(centred / tau.exp().unsqueeze(-1), dim=-1), metric
            )

        # Scores spread over `span` with spacing span/J: about t*J/span of them
        # participate, so t ~ span * target / J.  Increases with the target, as
        # it must.  The bracket expansion below is what is authoritative.
        tau0 = torch.log(scale) + math.log(max(target, 1.0 + 1e-6)) - math.log(j)
        lo, hi = tau0 - initial_log_range, tau0 + initial_log_range
        f_lo, f_hi = count(lo) - target, count(hi) - target
        for _ in range(40):
            bad_lo, bad_hi = f_lo > 0, f_hi < 0
            if not bool(bad_lo.any() or bad_hi.any()):
                break
            lo = torch.where(bad_lo, lo - 2.0, lo)
            hi = torch.where(bad_hi, hi + 2.0, hi)
            f_lo = torch.where(bad_lo, count(lo) - target, f_lo)
            f_hi = torch.where(bad_hi, count(hi) - target, f_hi)
        bracketed = (f_lo <= 0) & (f_hi >= 0)

        tau = 0.5 * (lo + hi)
        iters = torch.zeros_like(span)
        for step in range(max_iters):
            t = tau.exp()
            logits = centred / t.unsqueeze(-1)
            n, coeff = _count_and_coeff(logits, metric)
            f = n - target
            lo = torch.where(f <= 0, tau, lo)
            hi = torch.where(f > 0, tau, hi)
            done = f.abs() <= tol * target
            iters = torch.where(done, iters, torch.full_like(iters, step + 1.0))
            if bool(done.all()):
                break
            slope = -(coeff * logits).sum(-1)  # dN/dtau, since dlogits/dtau = -logits
            newton = tau - f / torch.where(slope.abs() < tiny, torch.full_like(slope, tiny), slope)
            usable = torch.isfinite(newton) & (newton > lo) & (newton < hi) & (slope > 0)
            step_tau = torch.where(usable, newton, 0.5 * (lo + hi))
            # freeze converged rows: at f == 0 the Newton point coincides with a
            # bracket endpoint, fails the strict test, and would otherwise be
            # kicked back to the midpoint
            tau = torch.where(done, tau, step_tau)

        t = tau.exp()
        t = torch.where(degenerate | ~bracketed, scale.clamp_min(tiny), t)
        status = torch.full_like(span, STATUS_OK, dtype=torch.long)
        status = torch.where(
            ~bracketed & (f_lo > 0), torch.full_like(status, STATUS_BELOW_RANGE), status
        )
        status = torch.where(
            ~bracketed & (f_hi < 0), torch.full_like(status, STATUS_ABOVE_RANGE), status
        )
        status = torch.where(degenerate, torch.full_like(status, STATUS_DEGENERATE), status)
        return t, {
            "temp_iters": iters,
            "temp_status": status,
            "n_eff_realized": score_softmax_count(r, t, metric),
        }


# --------------------------------------------------------------------------- #
# joint (b, tau) solver: exact gradient weights over a configurable subset
# --------------------------------------------------------------------------- #


def solve_joint_temperature(
    candidates: torch.Tensor,
    k: float,
    n_eff_target: float,
    b0: torch.Tensor,
    t0: torch.Tensor,
    metric: str = "ess",
    calibration: slice = slice(None),
    tol: float = 1e-5,
    budget_tol: float = 1e-6,
    max_iters: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Batched damped Newton on ``F1 = sum p - K`` and ``F2 = N_eff - target``.

    ``calibration`` selects which candidates enter ``F2``:

        slice(k, k + j)  -- one-sided true_gradient (inactive candidates only)
        slice(0, k + j)  -- two-sided (every candidate)

    ``F1`` always runs over the whole pool.  The Jacobian is the same in both
    cases; only the ``coeff``-dependent entries are restricted.  Initialized
    from the cheap ``score_softmax`` ``t0`` and its closed-form barrier ``b0``,
    which is normally already close.  Each iteration is elementwise work plus
    reductions over ``M`` and a 2x2 solve per row -- no large Jacobians.
    """
    with torch.no_grad():
        raw = candidates.float()
        m = raw.shape[-1]
        tiny = torch.finfo(raw.dtype).tiny
        target = float(n_eff_target)

        # Centre on r_K.  z = (r - b)/t is otherwise evaluated from raw scores,
        # so a large common offset (activations rarely sit around zero) burns
        # float32 mantissa that the small t then amplifies -- which pins the
        # residual at a noise floor well above the requested tolerance.
        centre = raw[..., min(max(int(k) - 1, 0), m - 1) : min(max(int(k), 1), m)]
        r = raw - centre
        b = (b0.float() - centre.squeeze(-1)).clone()
        tau = t0.float().clamp_min(tiny).log()
        iters = torch.zeros_like(b)
        done = torch.zeros_like(b, dtype=torch.bool)

        for step in range(max_iters):
            t = tau.exp()
            z = (r - b.unsqueeze(-1)) / t.unsqueeze(-1)
            az = z.abs()
            phi = 0.5 * torch.exp(-az)
            f1 = laplace_cdf(z).sum(-1) - k

            zc, azc = z[..., calibration], az[..., calibration]
            n, coeff = _count_and_coeff(-azc, metric)
            f2 = n - target

            done = (f1.abs() <= budget_tol * max(k, 1.0)) & (f2.abs() <= tol * target)
            iters = torch.where(done, iters, torch.full_like(iters, step + 1.0))
            if bool(done.all()):
                break

            j11 = -phi.sum(-1) / t
            j12 = -(phi * z).sum(-1)
            j21 = (coeff * torch.sign(zc)).sum(-1) / t
            j22 = (coeff * azc).sum(-1)
            det = j11 * j22 - j12 * j21
            det = torch.where(det.abs() < tiny, torch.full_like(det, tiny), det)

            db = (-f1 * j22 + f2 * j12) / det
            dtau = (j21 * f1 - j11 * f2) / det
            db = torch.clamp(db, -_B_CLIP * t, _B_CLIP * t)
            dtau = dtau.clamp(-_TAU_CLIP, _TAU_CLIP)
            live = (~done).float()
            b = b + 0.9 * db * live
            tau = tau + 0.9 * dtau * live

        t = tau.exp()
        b = b + centre.squeeze(-1)
        ok = done & torch.isfinite(b) & torch.isfinite(t)
        return b, t, ok, {"newton_iters": iters, "newton_failed": (~ok).float()}


# --------------------------------------------------------------------------- #
# reference solver: source of truth, and the fallback
# --------------------------------------------------------------------------- #


def solve_reference_temperature(
    candidates: torch.Tensor,
    k: float,
    n_eff_target: float,
    metric: str = "ess",
    calibration: slice = slice(None),
    grid: int = 96,
    refine: int = 60,
    log_lo: float = -9.0,
    log_hi: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Outer root-find over ``log t`` with the exact barrier solved inside.

    For each trial ``t``: solve ``b(t)`` in closed form, evaluate ``kappa`` at
    ``(b, t)``, restrict to the calibration subset, take ESS/entropy.  Slower
    than the joint Newton, and the source of truth for the tests.

    A grid scan rather than a plain bisection because the gradient-weight
    ``N_eff`` is not assumed to be globally monotone in ``t``: the scan finds a
    sign change wherever it is, and bisection only refines inside it.  The two
    endpoints double as the feasibility test: at a large ``t`` every weight is
    equal so ``N_eff`` is the subset size, and at a small ``t`` two-sided
    calibration bottoms out near 2 -- the two neurons straddling the boundary
    acquire equal density -- while one-sided bottoms out near 1.

    ``log_lo`` stops at ``span * e^-9`` rather than going arbitrarily low
    because further down the surrogate is numerically dead: every ``|z|`` leaves
    float range, the budget equation is flat so the barrier is only defined up
    to its plateau, and the weight softmax collapses onto whichever candidate
    rounding happens to favour -- which reads as ``N_eff -> 1`` and would make
    an unreachable target look reachable at a temperature that transmits no
    gradient at all.  The attainable floor is genuinely row-dependent, so it is
    reported per row rather than assumed.

    Returns ``(b, t, status)``.
    """
    with torch.no_grad():
        raw = candidates.float()
        m = raw.shape[-1]
        centre = raw[..., min(max(int(k) - 1, 0), m - 1) : min(max(int(k), 1), m)]
        r = raw - centre
        tiny = torch.finfo(r.dtype).tiny
        span = (r[..., :1] - r[..., -1:]).squeeze(-1)
        degenerate = span <= tiny
        scale = torch.where(degenerate, torch.ones_like(span), span)
        base = torch.log(scale.clamp_min(tiny))
        lo, hi = base + log_lo, base + log_hi

        def resid(tau: torch.Tensor) -> torch.Tensor:
            t = tau.exp()
            b = lapsum_barrier_sorted(r, k, t)
            return gradient_count(r, b, t, metric, calibration) - n_eff_target

        taus = torch.stack([lo + (hi - lo) * i / (grid - 1) for i in range(grid)], dim=0)
        vals = torch.stack([resid(taus[i]) for i in range(grid)], dim=0)

        # feasibility from the endpoints, before trying to bracket
        status = torch.full_like(span, STATUS_OK, dtype=torch.long)
        status = torch.where(
            vals[0] > 0, torch.full_like(status, STATUS_BELOW_RANGE), status
        )
        status = torch.where(
            vals[-1] < 0, torch.full_like(status, STATUS_ABOVE_RANGE), status
        )
        status = torch.where(degenerate, torch.full_like(status, STATUS_DEGENERATE), status)

        sign_change = (vals[:-1] <= 0) & (vals[1:] >= 0)
        any_change = sign_change.any(0)
        first = torch.where(any_change, sign_change.float().argmax(0), torch.zeros_like(lo).long())
        a = torch.gather(taus, 0, first.unsqueeze(0)).squeeze(0)
        c = torch.gather(taus, 0, (first + 1).clamp(max=grid - 1).unsqueeze(0)).squeeze(0)
        # no sign change: clamp to whichever endpoint is closest to the target
        a = torch.where(any_change, a, torch.where(vals[0] > 0, lo, hi))
        c = torch.where(any_change, c, torch.where(vals[0] > 0, lo, hi))
        for _ in range(refine):
            mid = 0.5 * (a + c)
            neg = resid(mid) <= 0
            a = torch.where(neg, mid, a)
            c = torch.where(neg, c, mid)
        tau = 0.5 * (a + c)
        t = tau.exp()
        return lapsum_barrier_sorted(r, k, t) + centre.squeeze(-1), t, status


def status_fractions(status: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-status row fractions, for logging."""
    return {
        f"status_{name}": (status == code).float().mean()
        for code, name in STATUS_NAMES.items()
        if code != STATUS_OK
    }
