"""DUST for the toy input gate (``topk.surrogate_mode: dust``): a zeroth-order,
reward-weighted-noise estimate of the loss gradient at the gate input,
alternated with ordinary hard Top-K steps.

The formulas are those of the main repository
(``wsparse.bottleneck.dust``, ``docs/dust-topk.md``) with one gate, no
sequence axis (a "token" is a sample, so ``gamma`` has nothing to discount and
must be 0) and the diagonal rescaling as the "encoder" of the gate input.

One DUST step, per batch ``(x, y)`` of ``B`` samples:

1. The clean forward (``train()`` mode, so the usage statistics are the
   training ones).  The gate records its input ``u = exp(g) * x`` ``[B, N]``.
2. ``K = dust_population`` draws.  Draw ``i`` adds independent Gaussian noise
   to the clean Top(K+J) pool ``C_t`` of every sample (ranked by ``|u|``; the
   noise acts on the signed value, before the absolute value)::

       u'_t = u_t + sigma_t a_t^(i),    a_t^(i) ~ N(0, I) on C_t, 0 off it,

   and the tail (gate, MLP, CE) is rerun from that input
   (``ToyMLP.forward_from_gate``, ``eval()`` mode, ``no_grad``, the CE in
   float32), giving the per-sample CE ``ell_t^(i)``.
3. With no later tokens the credit is the sample's own loss,
   ``R_t^(i) = ell_t^(i)``; the baseline is the mean over the draws,
   ``R~_t = (1 / K) sum_i R_t^(i)``, and the reward ``r_t^(i) = R~_t - R_t^(i)``.
4. The estimated output error (the reference's Eq. 2 with the ``1 / K``
   prefactor and the draw-mean baseline, divided by the number of samples
   ``B`` so that it estimates the gradient of the MEAN CE, the objective of
   the hard steps)::

       g^_t = -(1 / (K sigma_t B)) sum_i r_t^(i) a_t^(i),

   nonzero on ``C_t`` only.
5. ``dust_target``: the only parameter before the gate is ``g`` with
   ``d u_{t,n} / d g_n = exp(g_n) x_{t,n}``, so

   encoder   ``G^_g = sum_t g^_t * exp(g) * x_t``   (the diagonal of the outer
             product ``sum_t g^_t x_t^T``, times ``exp(g)``), added to ``g.grad``
             explicitly; nothing else moves in a DUST step.
   backprop  ``g^`` is backpropagated from ``u`` through the clean graph,
             which reaches ``g`` and nothing else: the same gradient, formed
             by autograd (a consistency check rather than a different method).

   ``dust_grad_scale`` multiplies ``g^`` before either target; ``ghat_rms`` is
   the unscaled estimate.  ``dust_optimizer: separate`` applies the DUST
   gradient with a second optimizer instance (train.py).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import OPTIMIZERS, SIGMA_MODES, TARGETS, TopKConfig  # noqa: F401  (re-exported)

#: version of the ``dust_state`` checkpoint payload
DUST_STATE_VERSION = 1

#: The tails' matmuls run in TF32 on CUDA (10-bit mantissa products, float32
#: accumulation; the main repository runs its tails in bf16, which is coarser).
#: The clean forward, the hard steps and validation keep full float32.  The
#: rounding error of a tail loss is about 1e-3, against reward differences of
#: order 1e-2 to 1e-1 at the shipped sigma; it halves the cost of the first
#: layer at hidden 1024.
TAIL_TF32 = True


# --------------------------------------------------------------------------- #
# schedule and generator
# --------------------------------------------------------------------------- #


def dust_step_kind(step: int, schedule: Sequence[int]) -> str:
    """``"hard"`` or ``"dust"`` for zero-based optimizer step ``step``.

    ``schedule = [n_hard, n_dust]``: every period of ``n_hard + n_dust`` steps
    begins with the hard steps.  A pure function of the step index.
    """
    n_hard, n_dust = int(schedule[0]), int(schedule[1])
    period = n_hard + n_dust
    if period < 1 or n_dust < 1:
        raise ValueError(f"dust_schedule needs n_hard >= 0 and n_dust >= 1, got {list(schedule)}")
    return "hard" if (int(step) % period) < n_hard else "dust"


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    """A dedicated RNG on ``device``: the noise never shares a stream with the
    data sampler or the initialization."""
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #


def pool_indices(u: torch.Tensor, m: int) -> torch.Tensor:
    """The clean Top(K+J) pool of every row by ``|u|``, ``[B, m]``, sorted."""
    return torch.topk(u.detach().abs(), int(m), dim=-1, largest=True, sorted=True).indices


def noise_scale(mode: str, sigma: float, u: torch.Tensor, cand_idx: torch.Tensor
                ) -> torch.Tensor:
    """Per-row noise width ``sigma_t``, float32 ``[B, 1]`` (detached;
    ``relative_rms`` scales by the RMS of the clean pool values of the row)."""
    if mode == "absolute":
        return torch.full(u.shape[:-1] + (1,), float(sigma), dtype=torch.float32,
                          device=u.device)
    if mode == "relative_rms":
        pool = u.detach().float().gather(-1, cand_idx)
        rms = pool.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
        return float(sigma) * rms
    raise ValueError(f"unknown dust_sigma_mode: {mode!r}")


def dust_estimate(u: torch.Tensor, cand_idx: Optional[torch.Tensor], sigma_t: torch.Tensor,
                  draws: int, loss_fn: Callable[[torch.Tensor], torch.Tensor],
                  generator: Optional[torch.Generator] = None, chunk: int = 1
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The reference's Eq. 2 on the pool, before the ``1 / B`` factor.

    ``u`` ``[B, N]`` (detached inside), ``cand_idx`` ``[B, M]`` the pool, or
    ``None`` when the pool is the whole input (then the noise is drawn in
    feature order and no scatter is formed; ``M = N``), ``sigma_t`` ``[B, 1]``
    float32.  ``loss_fn(u')`` maps a perturbed ``[c * B, N]`` tensor (``c``
    draws stacked along the batch axis, in ``u``'s dtype) to per-sample losses
    ``[c * B]``.  Returns ``(g^_pool, stats)`` with ``g^_pool`` ``[B, M]``
    float32 (in pool order, or feature order for the whole input)::

        g^_t = (1 / (K sigma_t)) sum_i (R_t^(i) - R~_t) a_t^(i)

    accumulated as ``S_Ra - R~ S_a`` so that no draw is stored.  ``stats``:
    ``ell_sum`` (the perturbed losses summed over draws and samples),
    ``reward_sq_sum`` (``sum_t sum_i (R_t^(i) - R~_t)^2``), ``draws``.
    """
    u = u.detach()
    B, N = u.shape
    M = N if cand_idx is None else cand_idx.shape[-1]
    dev = u.device
    f32 = torch.float32
    s_a = torch.zeros(B, M, dtype=f32, device=dev)
    s_ra = torch.zeros(B, M, dtype=f32, device=dev)
    r_sum = torch.zeros(B, dtype=f32, device=dev)
    r2_sum = torch.zeros(B, dtype=f32, device=dev)
    ell_sum = torch.zeros((), dtype=f32, device=dev)
    draws = int(draws)
    chunk = max(1, min(int(chunk), draws))
    uf = u.to(f32).unsqueeze(0)
    done = 0
    while done < draws:
        c = min(chunk, draws - done)
        a = torch.randn((c, B, M), generator=generator, dtype=f32, device=dev)
        if cand_idx is None:
            up = (uf + sigma_t.unsqueeze(0) * a).to(u.dtype).reshape(c * B, N)
        else:
            noise = torch.zeros((c, B, N), dtype=f32, device=dev)
            noise.scatter_(-1, cand_idx.unsqueeze(0).expand(c, B, M), sigma_t.unsqueeze(0) * a)
            up = (uf + noise).to(u.dtype).reshape(c * B, N)
            del noise
        r = loss_fn(up).reshape(c, B).to(f32)  # R_t^(i) = ell_t^(i): no later tokens
        del up
        s_a += a.sum(0)
        s_ra += (r.unsqueeze(-1) * a).sum(0)
        r_sum += r.sum(0)
        r2_sum += (r * r).sum(0)
        ell_sum += r.sum()
        done += c
        del a, r
    r_bar = r_sum / draws
    ghat = (s_ra - r_bar.unsqueeze(-1) * s_a) / (draws * sigma_t)
    reward_sq = (r2_sum - draws * r_bar * r_bar).clamp_min(0.0).sum()
    return ghat, {"ell_sum": ell_sum, "reward_sq_sum": reward_sq,
                  "draws": torch.tensor(float(draws), device=dev)}


# --------------------------------------------------------------------------- #
# trainer-side state and the step
# --------------------------------------------------------------------------- #


class DustTrainingState:
    """The DUST hyperparameters, the dedicated noise generator, and the count
    of completed DUST steps (the cadence of the cosine diagnostic)."""

    def __init__(self, cfg: TopKConfig, device: torch.device, seed: int):
        self.population = int(cfg.dust_population)
        self.n_bottlenecks = 1
        self.per_bottleneck = self.population  # one gate: K = dust_population
        if self.per_bottleneck < 2:
            raise ValueError("dust_population must be >= 2: the baseline is the mean over "
                             "the gate's own draws")
        self.sigma = float(cfg.dust_sigma)
        self.sigma_mode = str(cfg.dust_sigma_mode)
        if self.sigma_mode not in SIGMA_MODES:
            raise ValueError(f"unknown dust_sigma_mode: {self.sigma_mode!r}")
        self.gamma = float(cfg.dust_gamma)
        if self.gamma != 0.0:
            raise ValueError("dust_gamma must be 0 on MNIST (no later tokens)")
        self.schedule = [int(v) for v in cfg.dust_schedule]
        self.target = str(cfg.dust_target)
        if self.target not in TARGETS:
            raise ValueError(f"unknown dust_target: {self.target!r}")
        self.grad_scale = float(cfg.dust_grad_scale)
        if not (math.isfinite(self.grad_scale) and self.grad_scale > 0):
            raise ValueError("dust_grad_scale must be positive and finite")
        self.lr_scale = float(cfg.dust_lr_scale)
        if not (math.isfinite(self.lr_scale) and self.lr_scale > 0):
            raise ValueError("dust_lr_scale must be positive and finite")
        self.chunk = max(1, min(int(cfg.dust_chunk_draws), self.per_bottleneck))
        self.cosine_every = int(cfg.dust_cosine_every_steps)
        self.generator = make_generator(device, seed)
        self.device = device
        self.dust_steps = 0  # completed DUST optimizer steps

    def kind(self, step: int) -> str:
        return dust_step_kind(step, self.schedule)

    def cosine_due(self) -> bool:
        """Whether the DUST step about to run logs the cosine diagnostic."""
        return self.cosine_every > 0 and self.dust_steps % self.cosine_every == 0

    def describe(self) -> str:
        n_hard, n_dust = self.schedule
        return (f"schedule [{n_hard} hard, {n_dust} dust], population {self.population} "
                f"draws per batch, sigma {self.sigma:g} ({self.sigma_mode}), "
                f"gamma {self.gamma:g}, target {self.target}, grad scale {self.grad_scale:g}, "
                f"lr scale {self.lr_scale:g}, {self.chunk} draw(s) per tail forward"
                + (f", cosine every {self.cosine_every} dust steps"
                   if self.cosine_every > 0 else ""))

    def state_dict(self) -> Dict:
        return {"version": DUST_STATE_VERSION, "dust_steps": int(self.dust_steps),
                "generator_state": self.generator.get_state().cpu()}

    def load_state_dict(self, payload: Dict) -> None:
        if int(payload.get("version", -1)) != DUST_STATE_VERSION:
            raise ValueError(f"dust_state version {payload.get('version')!r} is not "
                             f"{DUST_STATE_VERSION}")
        self.dust_steps = int(payload["dust_steps"])
        self.generator.set_state(payload["generator_state"].cpu())


class DustRunningStats:
    """Sums of the per-step statistics between two logs."""

    KEYS = ("ce_clean", "ce_perturbed", "ce_gap", "reward_rms", "ghat_rms")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.micro = 0
        self.steps = 0
        self.sums: Dict[str, float] = {}
        self.cos_sum = 0.0
        self.cos_n = 0
        self.grad_norm_sum = 0.0

    def add(self, stats: Dict[str, float]) -> None:
        self.micro += 1
        for key in self.KEYS:
            self.sums[key] = self.sums.get(key, 0.0) + float(stats[key])
        if stats.get("cosine") is not None:
            self.cos_n += 1
            self.cos_sum += float(stats["cosine"])

    def add_step(self, grad_norm: float) -> None:
        self.steps += 1
        self.grad_norm_sum += float(grad_norm)

    def metrics(self) -> Dict[str, float]:
        out = {"dust/steps": float(self.steps)}
        if self.micro:
            for key in self.KEYS:
                out[f"dust/{key}"] = self.sums[key] / self.micro
        if self.steps:
            out["dust/grad_norm"] = self.grad_norm_sum / self.steps
        if self.cos_n:
            out["dust/cosine"] = self.cos_sum / self.cos_n
        return out


def dust_micro_step(model, x: torch.Tensor, y: torch.Tensor, state: DustTrainingState,
                    want_cosine: bool = False) -> Dict[str, float]:
    """One batch of a DUST step: the clean forward, ``K`` tails, and the
    gradient on ``g`` into ``.grad`` according to ``state.target``.  No CE
    backward.

    Returns detached floats: ``ce_clean`` and ``acc_clean`` (the clean
    forward's mean CE and accuracy, what the hard steps log as train/ce and
    train/acc), ``ce_perturbed`` (mean over draws and samples of the perturbed
    CE), ``ce_gap`` (perturbed - clean), ``reward_rms`` (RMS over draws and
    samples of ``R - R~``), ``ghat_rms`` (RMS over the pool entries of the
    UNSCALED ``g^``), and, with ``want_cosine``, ``cosine``: the cosine between
    the DUST gradient of ``g`` and the hard-backward gradient of the clean CE
    for ``g`` (else ``None``).
    """
    gate, rescale = model.gate, model.rescale
    if gate is None or rescale is None:
        raise RuntimeError("dust needs the gate and the diagonal rescaling installed")
    g = rescale.log_scale
    records: List[torch.Tensor] = []
    need_graph = state.target == "backprop" or want_cosine
    was_training = model.training
    model.train()
    with gate.capture(records), torch.set_grad_enabled(need_graph):
        logits, ce = model(x, y)
    if len(records) != 1:
        raise RuntimeError(f"dust: the clean forward recorded {len(records)} gate calls for 1 gate")
    u = records[0]
    B, N = u.shape
    n_valid = float(B)
    acc_clean = (logits.detach().argmax(-1) == y).float().mean()
    del logits
    true_grad: Optional[torch.Tensor] = None
    if want_cosine:
        (true_grad,) = torch.autograd.grad(ce, [g], retain_graph=(state.target == "backprop"))
    ce_clean = ce.detach().to(torch.float32)
    del ce

    f32 = torch.float32
    full_pool = gate.m >= N
    cand_idx = None if full_pool else pool_indices(u, gate.m)
    sigma_t = noise_scale(state.sigma_mode, state.sigma, u,
                          torch.arange(N, device=u.device).expand(B, N) if full_pool else cand_idx)
    model.eval()  # the tails: no usage statistics
    tf32_prior = torch.backends.cuda.matmul.allow_tf32
    if TAIL_TF32 and u.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        def loss_fn(up: torch.Tensor) -> torch.Tensor:
            c = up.shape[0] // B
            with torch.no_grad():
                lg = model.forward_from_gate(up)
            return F.cross_entropy(lg.float(), y.repeat(c), reduction="none")

        with torch.no_grad():
            ghat_pool, st = dust_estimate(u, cand_idx, sigma_t, state.per_bottleneck, loss_fn,
                                          state.generator, state.chunk)
            ghat_pool = ghat_pool / n_valid
            ghat_sq = ghat_pool.pow(2).sum()
            ghat_n = float(B * ghat_pool.shape[-1])
            # dust_grad_scale: the gradient that enters the optimizer
            if full_pool:
                ghat = ghat_pool * state.grad_scale
            else:
                ghat = torch.zeros(B, N, dtype=f32, device=u.device).scatter_(
                    -1, cand_idx, ghat_pool * state.grad_scale)
            # the "encoder" is the diagonal rescaling: d u_n / d g_n = exp(g_n) x_n
            g_est = (ghat * x.to(f32) * torch.exp(g.detach().to(f32))).sum(0)
    finally:
        model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = tf32_prior

    if state.target == "encoder":
        with torch.no_grad():
            est = g_est.to(g.dtype)
            g.grad = est if g.grad is None else g.grad + est
    else:
        torch.autograd.backward([u], [ghat.to(u.dtype)])
    del records, u

    cosine: Optional[float] = None
    if want_cosine and true_grad is not None:
        with torch.no_grad():
            denom = g_est.norm() * true_grad.to(f32).norm()
            cosine = float((g_est * true_grad.to(f32)).sum() / denom.clamp_min(torch.finfo(f32).tiny))
    draws_total = float(state.per_bottleneck)
    vals = torch.stack([ce_clean, acc_clean.to(f32), st["ell_sum"] / (draws_total * n_valid),
                        st["reward_sq_sum"], ghat_sq]).tolist()
    ce_c, acc_c, ce_p, rsq, gsq = vals
    return {"ce_clean": ce_c, "acc_clean": acc_c, "ce_perturbed": ce_p, "ce_gap": ce_p - ce_c,
            "reward_rms": math.sqrt(rsq / (draws_total * n_valid)),
            "ghat_rms": math.sqrt(gsq / ghat_n), "cosine": cosine}
