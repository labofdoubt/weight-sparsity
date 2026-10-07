"""Laplace-policy hard Top-K (``surrogate_mode="laplace_policy"``).

A gate row with signed encoder output ``z`` (``N`` features) and clean ranking
scores ``s`` (``|z|`` under abs_topk, ``z`` under topk) first takes its clean
Top-(K+J) candidate pool ``C``.  In a stochastic forward every candidate score
receives independent Laplace noise of width ``T``::

    r_i = u_i + eps_i,   eps_i ~ Laplace(0, T),   i in C,

where ``u`` is the clean candidate score, optionally centred within the pool
(``policy_center_scores``).  The K largest ``r_i`` form the support and their
ORIGINAL signed values ``z_i`` are transmitted unchanged: every stochastic
forward is a hard, exactly K-sparse forward.  Nothing outside the clean pool
can be selected and the support never has more than K members.

The objective is the expected sampled cross-entropy, trained with the
likelihood-ratio (score-function) estimator.  For one sequence ``b`` with mean
CE ``c_b`` and a detached baseline ``B``::

    L_backward = L_CE + [L_support - stopgrad(L_support)]
    L_support  = sum_b w_b stopgrad(c_b - B) sum_{a in b} gamma_a log rho_T(stopgrad(r_a) | u_a)
    log rho_T(r | u) = sum_i [ -log(2T) - |r_i - u_i| / T ]

so the numerical loss is the sampled CE while the gradient carries the
ordinary hard-mask value gradient plus the selection (density) gradient.  The
complete sampled ``r`` is detached in the density; ``u`` (and its optional
centring mean) stays differentiable; ``c_b - B``, ``T`` and ``gamma`` are
detached.

What is exact and what is deliberately approximate (keep in sync with the
docs):

1. With fixed candidate membership, an externally set absolute ``T``, a
   sample-independent baseline and ``gamma = 1`` the estimator is unbiased
   for the expected sampled CE at that step's temperature.
2. The deterministic Top-(K+J) pool is not smoothed: the construction
   differentiates selection within the current pool and the smooth
   computations on rank-stable regions only.
3. The relative widths (``relative_b``, ``relative_span``) use
   ``T = tau * a(theta)`` with ``a`` DETACHED: a frozen-scale partial
   gradient that omits the derivative through the activation-dependent
   width.  Not the full gradient of the activation-dependent-noise objective.
4. ``gamma != 1`` rescales the selection gradient only; it is a
   stabilization / ablation, not the gradient of the expected CE.

None of this truncates gradients between gates: a later gate's density term
backpropagates into earlier blocks through the sampled hard values, and the
earlier gate's own likelihood-ratio term accounts for changing its support.
There is no product of hand-written surrogate Jacobians along a carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

TEMPERATURE_MODES = ("absolute", "relative_b", "relative_span")
SCHEDULES = ("constant", "exponential")
SUPPORT_SCALE_MODES = ("constant", "effective_temperature", "scheduled_temperature")
BASELINES = ("none", "ema")

#: version of the optional ``policy_state`` checkpoint payload
POLICY_STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# schedule, width, support multiplier
# --------------------------------------------------------------------------- #


def scheduled_tau(step: int, tau0: float, schedule: str, tau_final: Optional[float] = None,
                  hold_steps: int = 0, anneal_steps: int = 0) -> float:
    """The scheduled parameter ``tau(t)`` at zero-based training step ``t``.

    ``constant``: ``tau0``.  ``exponential``::

        v(t)   = clip((t - h) / d, 0, 1)
        tau(t) = tau0 * (tau_f / tau0) ** v(t)

    with hold ``h`` and anneal duration ``d > 0``; ``tau_f`` is held after
    the anneal.  Validation and probes never advance ``t``; it is the same
    step index the LR schedule uses, so a resume recomputes it.
    """
    if schedule == "constant":
        return float(tau0)
    if schedule != "exponential":
        raise ValueError(f"unknown policy_temperature_schedule: {schedule!r}")
    if tau_final is None or anneal_steps <= 0:
        raise ValueError("the exponential schedule needs policy_temperature_final and "
                         "policy_temperature_anneal_steps > 0")
    v = min(1.0, max(0.0, (float(step) - float(hold_steps)) / float(anneal_steps)))
    return float(tau0) * (float(tau_final) / float(tau0)) ** v


def effective_width(mode: str, tau: float, score_c: torch.Tensor, k: int, j: int,
                    min_temperature: float
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row effective noise width ``T`` (DETACHED), shape ``score_c.shape[:-1] + (1,)``.

    ``score_c`` is the clean, sorted (descending) candidate pool of the row,
    UNCENTRED.  With zero-based sorted indexing the rank boundary ``s_(K+1)``
    is index ``k`` and the last pool member ``s_(K+J)`` is index ``k + j - 1``.

        absolute       T = max(T_min, tau)
        relative_b     T = max(T_min, tau * s_(K+1))
        relative_span  T = max(T_min, tau * (s_(K+1) - s_(K+J)))

    Returns ``(T, floor_binding, raw_scale)``: ``floor_binding`` marks rows
    where the floor replaced a smaller value, ``raw_scale`` is the activation
    scale before ``tau`` (1 in absolute mode).  The whole width is detached in
    every mode; this is the frozen-scale rule of the module docstring.
    """
    score_c = score_c.detach()
    if mode == "absolute":
        raw = torch.ones_like(score_c[..., :1])
    elif mode == "relative_b":
        raw = score_c[..., k:k + 1]
    elif mode == "relative_span":
        raw = score_c[..., k:k + 1] - score_c[..., k + j - 1:k + j]
    else:
        raise ValueError(f"unknown policy_temperature_mode: {mode!r}")
    t_raw = float(tau) * raw
    binding = t_raw < float(min_temperature)
    t = t_raw.clamp_min(float(min_temperature))
    return t.detach(), binding, raw.detach()


def support_multiplier(mode: str, gamma0: float, t_row: torch.Tensor, tau: float,
                       t_ref: float, gamma_max: Optional[float] = None) -> torch.Tensor:
    """Detached per-row multiplier ``gamma_a`` of the selection gradient.

        constant               gamma0
        effective_temperature  gamma0 * T_a / T_ref     (cancels the sampled 1/T_a)
        scheduled_temperature  gamma0 * tau(t) / tau_ref (cancels the annealed 1/tau,
                                                          keeps the inverse row scale)

    An optional ``gamma_max`` caps the result from above.  Same shape as
    ``t_row``.
    """
    if mode == "constant":
        g = torch.full_like(t_row, float(gamma0))
    elif mode == "effective_temperature":
        g = float(gamma0) * t_row.detach() / float(t_ref)
    elif mode == "scheduled_temperature":
        g = torch.full_like(t_row, float(gamma0) * float(tau) / float(t_ref))
    else:
        raise ValueError(f"unknown policy_support_scale_mode: {mode!r}")
    if gamma_max is not None:
        g = g.clamp_max(float(gamma_max))
    return g.detach()


# --------------------------------------------------------------------------- #
# sampling and density
# --------------------------------------------------------------------------- #


def at_least_float32(x: torch.Tensor) -> torch.Tensor:
    """``x`` in float32 unless it is already wider; keeps the autograd path."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """A dedicated RNG on ``device`` seeded with ``seed``.

    Raises instead of falling back to the global RNG when the backend has no
    device generator: the training noise must never share a stream with
    dropout, logging or validation.
    """
    try:
        gen = torch.Generator(device=device)
    except Exception as exc:  # pragma: no cover - backend without a generator
        raise RuntimeError(
            f"laplace_policy needs a dedicated torch.Generator on {device}, which this "
            f"backend does not provide ({exc}); use a cpu or cuda device") from exc
    gen.manual_seed(int(seed))
    return gen


def sample_laplace(shape, scale: torch.Tensor, generator: Optional[torch.Generator] = None,
                   dtype: torch.dtype = torch.float32,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    """Independent ``Laplace(0, scale)`` variates by the inverse CDF.

    ``v ~ U(-1/2, 1/2)``, ``eps = -scale * sign(v) * log(1 - 2|v|)``.  The
    uniform is drawn on ``[eps_m, 1 - eps_m)`` with ``eps_m`` the dtype's
    machine epsilon, so ``1 - 2|v| >= 2 eps_m > 0`` and no ``log(0)`` can
    occur.  That clamp cuts the two tails at probability ``~eps_m`` each, i.e.
    ``|eps| <= scale * log(1 / (2 eps_m))`` (about ``15.2 scale`` in float32):
    a machine-precision tail approximation, not a modelling choice.  ``scale``
    broadcasts against ``shape`` (per-row widths of shape ``[..., 1]``).
    """
    dev = device if device is not None else scale.device
    eps_m = torch.finfo(dtype).eps
    v = torch.rand(shape, generator=generator, dtype=dtype, device=dev)
    v = v * (1.0 - 2.0 * eps_m) + eps_m - 0.5
    return -scale.to(dtype) * torch.sign(v) * torch.log1p(-2.0 * v.abs())


def laplace_log_density(r: torch.Tensor, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``sum_i [-log(2T) - |r_i - u_i| / T]`` over the last dimension.

    ``r`` must arrive DETACHED (the realized noisy scores, all K+J of them,
    selected or not); differentiating through ``r = u + eps`` would cancel
    the ``u`` dependence and zero the location gradient.  ``t`` is detached
    as well, so ``-log(2T)`` carries no parameter gradient but is part of the
    actual density value.
    """
    return (-torch.log(2.0 * t) - (r - u).abs() / t).sum(-1)


# --------------------------------------------------------------------------- #
# forward-scoped settings and records
# --------------------------------------------------------------------------- #


@dataclass
class PolicyRecord:
    """One sampled gate invocation: its differentiable, gamma-weighted
    sequence log density and detached diagnostics.  ``gate`` identifies the
    module (several records per gate are legitimate: shared or repeated gates
    are recorded per instance, never overwritten)."""

    gate: object
    log_prob: torch.Tensor  # gamma * log rho, shape = input.shape[:-1]
    diag: Dict[str, torch.Tensor] = field(default_factory=dict)
    # only with ``keep_samples``: the realized noisy scores and the pool indices
    r: Optional[torch.Tensor] = None
    cand_idx: Optional[torch.Tensor] = None


class PolicyCollector:
    """Forward-scoped sink for :class:`PolicyRecord`.

    Created by ``TransformerLM.forward(return_loss_details=True)`` for ONE
    forward and dropped when it returns; never a module attribute that
    outlives the forward, so no graph is retained across micro-batches.
    """

    def __init__(self, keep_samples: bool = False):
        self.records: List[PolicyRecord] = []
        self.keep_samples = bool(keep_samples)
        self.closed = False

    def add(self, record: PolicyRecord) -> None:
        if self.closed:
            raise RuntimeError("PolicyCollector is closed: a gate recorded a policy term "
                               "outside the forward that opened the collector")
        self.records.append(record)

    def close(self) -> None:
        self.closed = True

    def sequence_log_prob(self) -> Optional[torch.Tensor]:
        """Per-sequence sum of every record, shape ``[B]``; ``None`` without records.

        Each record's log density has the gate input's leading shape
        (``[B, T]`` for a token-wise gate); everything after the batch axis
        is summed -- all positions, all gates, all placements of a sequence.
        """
        if not self.records:
            return None
        total = None
        for rec in self.records:
            lp = rec.log_prob
            lp = lp.reshape(lp.shape[0], -1).sum(-1) if lp.dim() > 1 else lp
            total = lp if total is None else total + lp
        return total


@dataclass
class PolicyForwardSettings:
    """Temporary per-forward request to the laplace_policy gates.

    ``sample``: ``True`` / ``False`` overrides the default (sample iff the
    gate is in ``train()`` mode); ``None`` keeps the default.  ``collector``
    receives the differentiable density terms (training only; ``None`` for a
    stochastic evaluation).  ``generator`` is the noise RNG (``None`` draws
    from the global RNG, which the trainer never does).  Applied with
    :func:`policy_forward`, which restores the previous settings in
    ``finally``, so nested calls and exceptions leave the gates clean.
    """

    sample: Optional[bool] = None
    collector: Optional[PolicyCollector] = None
    generator: Optional[torch.Generator] = None


class policy_forward:
    """``with policy_forward(gates, settings): ...`` -- set and restore
    ``gate._policy_settings`` on every laplace_policy gate."""

    def __init__(self, gates, settings: Optional[PolicyForwardSettings]):
        self.gates = [g for g in gates if getattr(g, "surrogate_mode", None) == "laplace_policy"]
        self.settings = settings
        self._prior: List[object] = []

    def __enter__(self):
        self._prior = [getattr(g, "_policy_settings", None) for g in self.gates]
        for g in self.gates:
            g._policy_settings = self.settings
        return self

    def __exit__(self, *exc):
        for g, prior in zip(self.gates, self._prior):
            g._policy_settings = prior
        return False


def policy_gates(model) -> list:
    """Every laplace_policy gate of ``model`` (the controller caches the list
    on the model as ``policy_gates``; a model assembled by hand is walked)."""
    cached = getattr(model, "policy_gates", None)
    if cached is not None:
        return list(cached)
    return [m for m in model.modules() if getattr(m, "surrogate_mode", None) == "laplace_policy"]


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def sequence_costs(ce_tokens: torch.Tensor, valid: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ce, seq_ce, seq_valid)`` from unreduced token CE and a validity mask.

    ``ce_tokens`` is float32 ``[B, T]`` with ignored targets contributing 0;
    ``valid`` is ``[B, T]`` bool.  ``seq_ce[b] = mean CE of sequence b`` (0 for
    a sequence without valid targets, which then gets zero weight), and
    ``ce = sum_b w_b seq_ce[b] = total CE / N_valid`` with ``w_b = n_b / N_valid``.
    A micro-batch without a single valid target is an error, not a NaN.
    """
    seq_valid = valid.sum(-1)
    n_valid = seq_valid.sum()
    if int(n_valid) == 0:
        raise ValueError("laplace_policy: the micro-batch has no valid targets "
                         "(every target is ignore_index)")
    seq_sum = ce_tokens.sum(-1)
    seq_ce = seq_sum / seq_valid.clamp_min(1).to(seq_sum.dtype)
    ce = seq_sum.sum() / n_valid.to(seq_sum.dtype)
    return ce, seq_ce, seq_valid


def policy_backward_loss(ce: torch.Tensor, seq_ce: torch.Tensor, seq_valid: torch.Tensor,
                         seq_log_prob: Optional[torch.Tensor], baseline: float
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The computational loss ``L_CE + [L_support - stopgrad(L_support)]``.

    ``seq_log_prob`` is the collector's per-sequence gamma-weighted log
    density (``None`` when no gate sampled: the loss is then the plain CE).
    Returns ``(loss, support, advantage)`` with ``support`` the detached value
    of ``L_support`` (a gradient diagnostic, not an objective) and
    ``advantage = stopgrad(seq_ce - B)`` per sequence.  The loss's VALUE equals
    ``ce``; only its gradient differs.
    """
    advantage = (seq_ce.detach() - float(baseline)).to(seq_ce.dtype)
    if seq_log_prob is None:
        return ce, ce.new_zeros(()), advantage
    w = seq_valid.to(seq_ce.dtype) / seq_valid.sum().to(seq_ce.dtype)
    support = (w * advantage * seq_log_prob.to(seq_ce.dtype)).sum()
    loss = ce + (support - support.detach())
    return loss, support.detach(), advantage


# --------------------------------------------------------------------------- #
# trainer-side state: baseline and noise RNG
# --------------------------------------------------------------------------- #


class PolicyTrainingState:
    """EMA baseline and the per-rank training noise generator.

    The baseline ``B`` is a scalar in mean-CE units.  It is held fixed for all
    micro-batches of an optimizer step and updated once, after their
    advantages have been formed, from DETACHED CE sums and valid-token counts
    gathered over every micro-batch (and every rank under DDP, so all ranks
    use the same ``B`` next step).  Validation and probes never touch it.
    """

    def __init__(self, mode: str, decay: float, initial: float, device: torch.device,
                 seed: int):
        if mode not in BASELINES:
            raise ValueError(f"unknown policy_baseline: {mode!r}")
        self.mode = mode
        self.decay = float(decay)
        self.baseline = 0.0 if mode == "none" else float(initial)
        self.updates = 0
        self.generator = make_generator(device, seed)
        self.device = device
        self._ce_sum = 0.0
        self._n_sum = 0

    # -- per step ---------------------------------------------------------- #
    def accumulate(self, ce_sum: float, n_valid: int) -> None:
        """Record one micro-batch's total CE and valid-token count (detached)."""
        self._ce_sum += float(ce_sum)
        self._n_sum += int(n_valid)

    def finish_step(self, world: int = 1) -> float:
        """All-reduce the step's CE statistics, update the EMA once, reset.

        Called on EVERY rank (it is a collective under DDP), outside any
        rank-0-only block.  Returns the step's global mean CE.
        """
        if world > 1:
            import torch.distributed as dist
            dev = self.device if self.device.type == "cuda" else torch.device("cpu")
            stats = torch.tensor([self._ce_sum, float(self._n_sum)], dtype=torch.float64,
                                 device=dev)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            ce_sum, n_sum = float(stats[0]), int(round(float(stats[1])))
        else:
            ce_sum, n_sum = self._ce_sum, self._n_sum
        self._ce_sum, self._n_sum = 0.0, 0
        if n_sum == 0:
            return float("nan")
        mean_ce = ce_sum / n_sum
        if self.mode == "ema" and math.isfinite(mean_ce):
            self.baseline = self.decay * self.baseline + (1.0 - self.decay) * mean_ce
            self.updates += 1
        return mean_ce

    # -- checkpointing ----------------------------------------------------- #
    def state_dict(self, generator_states: Optional[List[torch.Tensor]] = None) -> Dict:
        """Versioned payload.  ``generator_states`` holds every rank's RNG state
        (gathered by the caller on all ranks); ``None`` means this rank's only."""
        states = (generator_states if generator_states is not None
                  else [self.generator.get_state().cpu()])
        return {"version": POLICY_STATE_VERSION, "baseline_mode": self.mode,
                "baseline": float(self.baseline), "updates": int(self.updates),
                "decay": self.decay, "world_size": len(states),
                "generator_states": [s.cpu() for s in states]}

    def load_state_dict(self, payload: Dict, rank: int, world: int) -> None:
        if int(payload.get("version", -1)) != POLICY_STATE_VERSION:
            raise ValueError(f"policy_state version {payload.get('version')!r} is not "
                             f"{POLICY_STATE_VERSION}")
        if payload.get("baseline_mode") != self.mode:
            raise ValueError(f"policy_state was saved with policy_baseline="
                             f"{payload.get('baseline_mode')!r}, the config says {self.mode!r}")
        if int(payload["world_size"]) != int(world):
            raise ValueError(
                f"policy_state holds {payload['world_size']} per-rank noise generator states "
                f"but this run has world size {world}; resuming a laplace_policy run with "
                "a different number of processes is not supported (no reseeding policy "
                "is defined in this version)")
        self.baseline = float(payload["baseline"])
        self.updates = int(payload["updates"])
        self.generator.set_state(payload["generator_states"][rank].cpu())
