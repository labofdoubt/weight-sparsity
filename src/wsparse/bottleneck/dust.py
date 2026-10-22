"""DUST for the activation bottleneck (``surrogate_mode="dust"``): a
zeroth-order, reward-weighted-noise estimate of the loss gradient at the
Top-K pre-activations, alternated with ordinary hard Top-K steps.

Reference: "Dust" (https://qlabs.sh/research/dust) -- node perturbation at
every token, the population evaluated in batched forward passes, the layer's
output error estimated from the rewards and fed to the same outer product
backprop would form.  Here the perturbed nodes are the gate inputs of the
installed bottlenecks only; everything else is trained by the ordinary hard
steps of the alternation.

One DUST step, per micro-batch ``(x, y)`` of ``B`` sequences and ``T`` tokens:

1. The clean forward (``train()`` mode, so the usage statistics are the
   training ones).  Every installed bottleneck ``l`` records its gate input
   ``z_l`` (``[B, T, N]``; under code_residual, for a block >= 1, that is the
   carried code plus the encoded update) and the input of its encoder.
2. For each bottleneck, ``K_l = dust_population / L`` draws.  Draw ``i`` adds
   independent Gaussian noise to the clean Top(K+J) pool ``C_t`` of every
   token (ranked by ``|z|`` under abs_topk, ``z`` under topk; the noise acts
   on the signed pre-activation, before the absolute value)::

       z'_t = z_t + sigma_t a_t^(i),    a_t^(i) ~ N(0, I) on C_t, 0 off it,

   and the forward is rerun from that gate on
   (``TransformerLM.forward_from_gate``, ``eval()`` mode, ``no_grad``, the
   unembedding and the CE in float32), giving the per-token CE
   ``ell_s^(i)`` (0 at ignored targets).
3. Discounted credit over later tokens, the mean over the bottleneck's own
   draws as the baseline, and the reward::

       R_t^(i) = sum_{s >= t} gamma^(s - t) ell_s^(i)
       R~_t    = (1 / K_l) sum_i R_t^(i)
       r_t^(i) = R~_t - R_t^(i)

4. The estimated output error (the reference's Eq. 2 with the ``1 / K_l``
   prefactor and the draw-mean baseline; divided by the number of valid
   targets ``N_valid`` so that it estimates the gradient of the MEAN token
   CE, the objective of the hard steps)::

       g^_t = -(1 / (K_l sigma_t N_valid)) sum_i r_t^(i) a_t^(i),

   which is nonzero on ``C_t`` only.  With ``gamma = 0`` a token's noise is
   credited with its own loss change alone.
5. ``dust_target``:

   encoder   bottleneck ``l``'s encoder weight receives
             ``G^_W = scale * sum_t g^_t x_t^T`` (``x_t`` the encoder input of
             the clean forward; ``scale`` = code_residual_scale for a
             code-residual update, else 1) and its bias ``scale * sum_t g^_t``;
             no other parameter moves in a DUST step.
   backprop  ``g^_l`` is backpropagated from ``z_l`` through the clean graph:
             every parameter preceding ``z_l`` -- earlier blocks, earlier
             bottlenecks through their hard masks and the carry, the
             embedding -- accumulates it; the gates on the way pass the hard
             mask, as in a hard step.

   ``dust_grad_scale`` multiplies ``g^`` before either target, so the whole
   DUST gradient scales with it; the ``ghat_rms`` statistic is the unscaled
   estimate.  ``dust_optimizer: separate`` applies the DUST gradient with a
   second optimizer instance (own Adam moments and step count), so neither
   step kind's gradient statistics set the other's step size (train.py).

Deliberate choices, kept in sync with ``docs/dust-topk.md``:

- The baseline is the mean over the SAME draws (the reference's).  For a
  loss linear in the perturbation this scales the expectation by
  ``1 - 1 / K_l``; the ``1 / K_l`` prefactor is kept as in the reference.
- The pool is the clean one.  A perturbation can swap pool members in or
  out of the hard support at this and at every later gate: that selection
  effect is in the reward, which is the point of a zeroth-order estimate.
- The tails run the body in the ambient precision and the head in float32.
  Under bf16 the perturbed pre-activations are rounded to bf16 before the
  gate, so ``sigma`` should stay well above the bf16 resolution of ``z``
  (about ``4e-3 |z|``).
- Single process only: the DUST step does not go through DDP's reducer.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .laplace_policy import make_generator

SIGMA_MODES = ("absolute", "relative_rms")
TARGETS = ("encoder", "backprop")
#: which optimizer applies the DUST gradient (see ActivationBottleneckConfig)
OPTIMIZERS = ("shared", "separate")

#: version of the optional ``dust_state`` checkpoint payload
DUST_STATE_VERSION = 1


# --------------------------------------------------------------------------- #
# schedule and population
# --------------------------------------------------------------------------- #


def dust_step_kind(step: int, schedule: Sequence[int]) -> str:
    """``"hard"`` or ``"dust"`` for zero-based optimizer step ``step``.

    ``schedule = [n_hard, n_dust]``: every period of ``n_hard + n_dust`` steps
    begins with the hard steps.  Pure function of the step index, so a resume
    continues the pattern.
    """
    n_hard, n_dust = int(schedule[0]), int(schedule[1])
    period = n_hard + n_dust
    if period < 1 or n_dust < 1:
        raise ValueError(f"dust_schedule needs n_hard >= 0 and n_dust >= 1, got {list(schedule)}")
    return "hard" if (int(step) % period) < n_hard else "dust"


def draws_per_bottleneck(population: int, n_bottlenecks: int) -> int:
    """``K_l = dust_population / L``, which must be an integer >= 2."""
    population, n_bottlenecks = int(population), int(n_bottlenecks)
    if n_bottlenecks < 1:
        raise ValueError("dust needs at least one installed bottleneck")
    if population % n_bottlenecks != 0:
        raise ValueError(
            f"dust_population={population} must be a multiple of the number of installed "
            f"bottlenecks ({n_bottlenecks}): every bottleneck gets the same share")
    per = population // n_bottlenecks
    if per < 2:
        raise ValueError(
            f"dust_population={population} gives {per} draw(s) per bottleneck "
            f"({n_bottlenecks} bottlenecks); the baseline is the mean over a bottleneck's "
            "own draws, so at least 2 are needed")
    return per


# --------------------------------------------------------------------------- #
# capture of the clean forward
# --------------------------------------------------------------------------- #


@dataclass
class DustRecord:
    """One gate call of the clean forward: ``z`` the gate input (the
    perturbed pre-activation), ``enc_input`` the encoder's input, with
    ``d z / d(W enc_input) = scale``.  ``z`` keeps its graph when the forward
    ran with grad enabled."""

    module: object
    enc_input: torch.Tensor
    z: torch.Tensor
    scale: float = 1.0


@contextmanager
def dust_capture(modules: Sequence[object], records: List[DustRecord]) -> Iterator[None]:
    """Install ``records`` as the capture list of every module for one forward."""
    prior = [getattr(m, "_dust_capture", None) for m in modules]
    for m in modules:
        m._dust_capture = records
    try:
        yield
    finally:
        for m, p in zip(modules, prior):
            m._dust_capture = p


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #


def token_losses(logits: torch.Tensor, targets: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(ell, valid)``: float32 per-token CE ``[B, T]`` (0 at ignored targets)
    and the validity mask.  The CE is formed from float32 logits."""
    B, T, V = logits.shape
    flat = F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1),
                           ignore_index=-100, reduction="none")
    valid = targets != -100
    ell = torch.where(valid, flat.view(B, T), flat.new_zeros(()))
    return ell, valid


def discounted_returns(ell: torch.Tensor, gamma: float) -> torch.Tensor:
    """``R_t = sum_{s >= t} gamma^(s - t) ell_s`` along the last axis.

    ``gamma = 0`` returns ``ell``; otherwise one float64 matmul with the
    ``[T, T]`` discount matrix (exact for ``gamma = 1``, the reverse cumsum).
    """
    gamma = float(gamma)
    if gamma == 0.0:
        return ell
    T = ell.shape[-1]
    idx = torch.arange(T, device=ell.device)
    d = (idx[None, :] - idx[:, None]).to(torch.float64)  # d[t, s] = s - t
    G = torch.where(d >= 0, torch.pow(torch.tensor(gamma, dtype=torch.float64,
                                                   device=ell.device), d),
                    torch.zeros((), dtype=torch.float64, device=ell.device))
    return (ell.to(torch.float64) @ G.transpose(0, 1)).to(ell.dtype)


def pool_indices(z: torch.Tensor, selection_mode: str, m: int) -> torch.Tensor:
    """The clean Top(K+J) pool of every row, ``[..., m]``, by the gate's scores."""
    scores = z.detach().abs() if selection_mode == "abs_topk" else z.detach()
    return torch.topk(scores, int(m), dim=-1, largest=True, sorted=True).indices


def noise_scale(mode: str, sigma: float, z: torch.Tensor, cand_idx: torch.Tensor
                ) -> torch.Tensor:
    """Per-row noise width ``sigma_t``, float32, shape ``z.shape[:-1] + (1,)``
    (DETACHED; ``relative_rms`` scales by the RMS of the clean pool values)."""
    if mode == "absolute":
        return torch.full(z.shape[:-1] + (1,), float(sigma), dtype=torch.float32,
                          device=z.device)
    if mode == "relative_rms":
        pool = z.detach().float().gather(-1, cand_idx)
        rms = pool.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
        return float(sigma) * rms
    raise ValueError(f"unknown dust_sigma_mode: {mode!r}")


def dust_estimate(z: torch.Tensor, cand_idx: torch.Tensor, sigma_t: torch.Tensor,
                  draws: int, gamma: float,
                  loss_fn: Callable[[torch.Tensor], torch.Tensor],
                  generator: Optional[torch.Generator] = None, chunk: int = 1
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The reference's Eq. 2 on the pool, before the ``1 / N_valid`` factor.

    ``z`` ``[B, T, N]`` (detached inside), ``cand_idx`` ``[B, T, M]`` the pool,
    ``sigma_t`` ``[B, T, 1]`` float32.  ``loss_fn(z')`` maps a perturbed
    ``[c * B, T, N]`` tensor (``c`` draws stacked along the batch axis, in
    ``z``'s dtype) to per-token losses ``[c * B, T]``.  Returns
    ``(g^_pool, stats)`` with ``g^_pool`` ``[B, T, M]`` float32::

        g^_t = (1 / (K sigma_t)) sum_i (R_t^(i) - R~_t) a_t^(i)

    (``= -(1 / (K sigma_t)) sum_i r_t^(i) a_t^(i)``), accumulated as
    ``S_Ra - R~ S_a`` so that no draw is stored.  ``stats``: ``ell_sum`` (the
    perturbed losses summed over draws and tokens), ``reward_sq_sum``
    (``sum_t sum_i (R_t^(i) - R~_t)^2``), ``draws``, and ``snr`` ``[B, T]``,
    the signal-to-noise ratio of the population-averaged estimate per token::

        z_t^(i) = -r_t^(i) a_t^(i) / sigma_t          (one draw's estimate)
        mu_t    = (1 / K) sum_i z_t^(i)               (= g^_t)
        nu_t^2  = (1 / (K - 1)) sum_i ||z_t^(i) - mu_t||^2
        SNR_t   = sqrt(K) ||mu_t|| / nu_t

    with ``sum_i ||z_t^(i) - mu_t||^2 = sum_i ||z_t^(i)||^2 - K ||mu_t||^2``
    and ``sum_i ||z_t^(i)||^2`` formed from the running sums of
    ``R^2 ||a||^2``, ``R ||a||^2`` and ``||a||^2``; 0 where the draws agree.
    """
    z = z.detach()
    B, T, N = z.shape
    M = cand_idx.shape[-1]
    dev = z.device
    f32 = torch.float32
    s_a = torch.zeros(B, T, M, dtype=f32, device=dev)
    s_ra = torch.zeros(B, T, M, dtype=f32, device=dev)
    r_sum = torch.zeros(B, T, dtype=f32, device=dev)
    r2_sum = torch.zeros(B, T, dtype=f32, device=dev)
    a2_sum = torch.zeros(B, T, dtype=f32, device=dev)      # sum_i ||a_i||^2
    ra2_sum = torch.zeros(B, T, dtype=f32, device=dev)     # sum_i R_i ||a_i||^2
    r2a2_sum = torch.zeros(B, T, dtype=f32, device=dev)    # sum_i R_i^2 ||a_i||^2
    ell_sum = torch.zeros((), dtype=f32, device=dev)
    draws = int(draws)
    chunk = max(1, min(int(chunk), draws))
    zf = z.to(f32).unsqueeze(0)
    done = 0
    while done < draws:
        c = min(chunk, draws - done)
        a = torch.randn((c, B, T, M), generator=generator, dtype=f32, device=dev)
        noise = torch.zeros((c, B, T, N), dtype=f32, device=dev)
        noise.scatter_(-1, cand_idx.unsqueeze(0).expand(c, B, T, M), sigma_t.unsqueeze(0) * a)
        zp = (zf + noise).to(z.dtype).reshape(c * B, T, N)
        del noise
        ell = loss_fn(zp).reshape(c, B, T).to(f32)
        del zp
        r = discounted_returns(ell, gamma).to(f32)
        s_a += a.sum(0)
        s_ra += (r.unsqueeze(-1) * a).sum(0)
        r_sum += r.sum(0)
        r2_sum += (r * r).sum(0)
        a2 = (a * a).sum(-1)
        a2_sum += a2.sum(0)
        ra2_sum += (r * a2).sum(0)
        r2a2_sum += (r * r * a2).sum(0)
        ell_sum += ell.sum()
        done += c
        del a, ell, r
    r_bar = r_sum / draws
    ghat = (s_ra - r_bar.unsqueeze(-1) * s_a) / (draws * sigma_t)
    reward_sq = (r2_sum - draws * r_bar * r_bar).clamp_min(0.0).sum()
    # SNR of the population-averaged estimate, per token
    sig2 = sigma_t.squeeze(-1).pow(2)
    mu2 = ghat.pow(2).sum(-1)
    z2_sum = (r2a2_sum - 2.0 * r_bar * ra2_sum + r_bar * r_bar * a2_sum) / sig2
    nu2 = (z2_sum - draws * mu2).clamp_min(0.0) / max(draws - 1, 1)
    snr = torch.where(nu2 > 0, math.sqrt(draws) * mu2.sqrt() / nu2.sqrt().clamp_min(
        torch.finfo(f32).tiny), torch.zeros_like(mu2))
    return ghat, {"ell_sum": ell_sum, "reward_sq_sum": reward_sq,
                  "draws": torch.tensor(float(draws), device=dev), "snr": snr}


# --------------------------------------------------------------------------- #
# trainer-side state and the step
# --------------------------------------------------------------------------- #


class DustTrainingState:
    """The DUST hyperparameters resolved against the installed bottlenecks,
    the dedicated noise generator, and the count of completed DUST steps
    (the cadence of the cosine diagnostic).  Validation never touches it."""

    def __init__(self, cfg, n_bottlenecks: int, device: torch.device, seed: int):
        self.population = int(cfg.dust_population)
        self.n_bottlenecks = int(n_bottlenecks)
        self.per_bottleneck = draws_per_bottleneck(self.population, self.n_bottlenecks)
        self.sigma = float(cfg.dust_sigma)
        self.sigma_mode = str(cfg.dust_sigma_mode)
        self.gamma = float(cfg.dust_gamma)
        self.schedule = [int(v) for v in cfg.dust_schedule]
        self.target = str(cfg.dust_target)
        if self.target not in TARGETS:
            raise ValueError(f"unknown dust_target: {self.target!r}")
        self.grad_scale = float(getattr(cfg, "dust_grad_scale", 1.0))
        if not (math.isfinite(self.grad_scale) and self.grad_scale > 0):
            raise ValueError("dust_grad_scale must be positive and finite")
        self.lr_scale = float(getattr(cfg, "dust_lr_scale", 1.0))
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
                f"= {self.n_bottlenecks} x {self.per_bottleneck} draws per micro-batch, "
                f"sigma {self.sigma:g} ({self.sigma_mode}), gamma {self.gamma:g}, "
                f"target {self.target}, grad scale {self.grad_scale:g}, "
                f"lr scale {self.lr_scale:g}, "
                f"{self.chunk} draw(s) per tail forward"
                + (f", cosine every {self.cosine_every} dust steps"
                   if self.cosine_every > 0 else ""))

    # -- checkpointing ----------------------------------------------------- #
    def state_dict(self) -> Dict:
        return {"version": DUST_STATE_VERSION, "dust_steps": int(self.dust_steps),
                "world_size": 1, "generator_state": self.generator.get_state().cpu()}

    def load_state_dict(self, payload: Dict) -> None:
        if int(payload.get("version", -1)) != DUST_STATE_VERSION:
            raise ValueError(f"dust_state version {payload.get('version')!r} is not "
                             f"{DUST_STATE_VERSION}")
        if int(payload.get("world_size", 1)) != 1:
            raise ValueError("dust_state holds more than one noise generator state; "
                             "dust is single-process only")
        self.dust_steps = int(payload["dust_steps"])
        self.generator.set_state(payload["generator_state"].cpu())


class DustRunningStats:
    """Sums of the per-micro-batch statistics between two logs."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.micro = 0
        self.steps = 0
        self.sums: Dict[str, float] = {}
        self.snr_sums: Dict[str, float] = {}
        self.cos_sums: Dict[str, float] = {}
        self.cos_n = 0
        self.grad_norm_sum = 0.0

    def add(self, stats: Dict[str, float]) -> None:
        self.micro += 1
        for key in ("ce_clean", "ce_perturbed", "ce_gap", "reward_rms", "ghat_rms", "snr"):
            self.sums[key] = self.sums.get(key, 0.0) + float(stats[key])
        for name, value in stats.get("snr_layers", {}).items():
            self.snr_sums[name] = self.snr_sums.get(name, 0.0) + float(value)
        cos = stats.get("cosine")
        if cos:
            self.cos_n += 1
            for name, value in cos.items():
                self.cos_sums[name] = self.cos_sums.get(name, 0.0) + float(value)

    def add_step(self, grad_norm: float) -> None:
        self.steps += 1
        self.grad_norm_sum += float(grad_norm)

    def metrics(self) -> Dict[str, float]:
        out = {"dust/steps": float(self.steps)}
        if self.micro:
            for key in ("ce_perturbed", "ce_gap", "reward_rms", "ghat_rms", "snr"):
                out[f"dust/{key}"] = self.sums[key] / self.micro
            out["dust/ce_clean"] = self.sums["ce_clean"] / self.micro
            for name, total in self.snr_sums.items():
                out[f"dust_snr/{name}"] = total / self.micro
        if self.steps:
            out["dust/grad_norm"] = self.grad_norm_sum / self.steps
        if self.cos_n:
            vals = []
            for name, total in self.cos_sums.items():
                out[f"dust_cosine/{name}"] = total / self.cos_n
                vals.append(total / self.cos_n)
            out["dust/cosine"] = sum(vals) / len(vals)
        return out


def _unique_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def dust_micro_step(model, controller, x: torch.Tensor, y: torch.Tensor,
                    state: DustTrainingState, device: torch.device, dtype: torch.dtype,
                    accum: int = 1, want_cosine: bool = False) -> Dict[str, float]:
    """One micro-batch of a DUST step on the UNWRAPPED ``model``: the clean
    forward, ``K_l`` tails per bottleneck, and the gradient accumulation into
    ``.grad`` (scaled by ``1 / accum``) according to ``state.target``.

    Returns detached floats: ``ce_clean`` (the clean mean CE, what the hard
    steps log as train/ce), ``ce_perturbed`` (mean over draws, bottlenecks
    and valid tokens of the perturbed CE), ``ce_gap`` (perturbed - clean),
    ``reward_rms`` (RMS over draws and valid tokens of ``R - R~``),
    ``ghat_rms`` (RMS over the pool entries of valid tokens of ``g^``), ``snr``
    (the per-token SNR of :func:`dust_estimate` averaged over valid tokens and
    bottlenecks) with ``snr_layers`` per layer label, and, with
    ``want_cosine``, ``cosine``: per distinct encoder weight (keyed by the
    first layer label holding it) the cosine between the DUST outer product
    and the hard-backward gradient of the clean CE.
    """
    from ..utils import autocast_context

    layers = list(controller.layers)
    modules = [m for _, m in layers]
    label_of = {}
    layer_label = {id(m): label for label, m in layers}
    for label, m in layers:
        label_of.setdefault(id(m.in_proj.weight), label)
    L = len(modules)
    per_l = state.per_bottleneck
    sel_mode = controller.cfg.selection_mode
    records: List[DustRecord] = []
    need_graph = state.target == "backprop" or want_cosine
    was_training = model.training
    model.train()
    with dust_capture(modules, records), torch.set_grad_enabled(need_graph), \
            autocast_context(device, dtype):
        logits, ce = model(x, y)
    if len(records) != L:
        raise RuntimeError(f"dust: the clean forward recorded {len(records)} gate calls for "
                           f"{L} installed bottlenecks")
    _, valid = token_losses(logits.detach(), y)
    del logits
    n_valid = valid.sum().to(torch.float32)
    if float(n_valid) == 0:
        raise ValueError("dust: the micro-batch has no valid targets")
    enc_weights = _unique_params([m.in_proj.weight for m in modules])
    true_grads: Dict[int, torch.Tensor] = {}
    if want_cosine:
        grads = torch.autograd.grad(ce, enc_weights, retain_graph=(state.target == "backprop"),
                                    allow_unused=True)
        true_grads = {id(w): g for w, g in zip(enc_weights, grads)}
    ce_clean = ce.detach().to(torch.float32)
    del ce

    f32 = torch.float32
    outer: Dict[int, torch.Tensor] = {}
    outer_bias: Dict[int, torch.Tensor] = {}
    roots: List[torch.Tensor] = []
    root_grads: List[torch.Tensor] = []
    ell_sum = torch.zeros((), dtype=f32, device=device)
    reward_sq = torch.zeros((), dtype=f32, device=device)
    ghat_sq = torch.zeros((), dtype=f32, device=device)
    ghat_n = torch.zeros((), dtype=f32, device=device)
    snr_layers: Dict[str, torch.Tensor] = {}
    model.eval()  # the tails: no dropout, no usage statistics, no diagnostics
    try:
        for rec in records:
            z = rec.z.detach()
            B, T, N = z.shape
            block_index, placement = rec.module._site
            gate = rec.module.gate
            cand_idx = pool_indices(z, sel_mode, gate.m)
            sigma_t = noise_scale(state.sigma_mode, state.sigma, z, cand_idx)

            def loss_fn(zp: torch.Tensor, _b=block_index, _p=placement) -> torch.Tensor:
                c = zp.shape[0] // B
                with torch.no_grad(), autocast_context(device, dtype):
                    lg = model.forward_from_gate(_b, _p, zp, head_dtype=torch.float32)
                ell, _ = token_losses(lg, y.repeat(c, 1))
                return ell

            with torch.no_grad():
                ghat_pool, st = dust_estimate(z, cand_idx, sigma_t, per_l, state.gamma, loss_fn,
                                              state.generator, state.chunk)
                ghat_pool = ghat_pool / n_valid
                ell_sum += st["ell_sum"]
                reward_sq += st["reward_sq_sum"]
                ghat_sq += (ghat_pool.pow(2) * valid.unsqueeze(-1)).sum()  # unscaled
                ghat_n += n_valid * ghat_pool.shape[-1]
                snr_layers[layer_label[id(rec.module)]] = (st["snr"] * valid).sum() / n_valid
                # dust_grad_scale: the gradient that enters the optimizer
                ghat = torch.zeros(B, T, N, dtype=f32, device=device).scatter_(
                    -1, cand_idx, ghat_pool * state.grad_scale)
                if state.target == "encoder" or want_cosine:
                    w = rec.module.in_proj.weight
                    g_w = torch.einsum("btn,btd->nd", ghat,
                                       rec.enc_input.detach().to(f32)) * rec.scale
                    outer[id(w)] = outer[id(w)] + g_w if id(w) in outer else g_w
                    b = rec.module.in_proj.bias
                    if b is not None:
                        g_b = ghat.sum((0, 1)) * rec.scale
                        outer_bias[id(b)] = (outer_bias[id(b)] + g_b
                                             if id(b) in outer_bias else g_b)
            if state.target == "backprop":
                roots.append(rec.z)
                root_grads.append((ghat / accum).to(rec.z.dtype))
            del ghat, ghat_pool
    finally:
        model.train(was_training)

    if state.target == "encoder":
        # each distinct parameter once: a shared encoder holds the sum of every
        # bottleneck's outer product already
        with torch.no_grad():
            weights = _unique_params([m.in_proj.weight for m in modules])
            biases = _unique_params([m.in_proj.bias for m in modules
                                     if m.in_proj.bias is not None])
            for p, store in [(w, outer) for w in weights] + [(b, outer_bias) for b in biases]:
                if id(p) not in store:
                    continue
                g = store[id(p)].to(p.dtype) / accum
                p.grad = g if p.grad is None else p.grad + g
    else:
        torch.autograd.backward(roots, root_grads)
    del roots, root_grads, records

    cosine: Dict[str, float] = {}
    if want_cosine:
        with torch.no_grad():
            for w in enc_weights:
                est, ref = outer.get(id(w)), true_grads.get(id(w))
                if est is None or ref is None:
                    continue
                denom = est.norm() * ref.to(f32).norm()
                cos = (est * ref.to(f32)).sum() / denom.clamp_min(torch.finfo(f32).tiny)
                cosine[label_of[id(w)]] = float(cos)
    draws_total = float(per_l * L)
    labels = list(snr_layers)
    vals = torch.stack([ce_clean, ell_sum / (draws_total * n_valid), reward_sq,
                        ghat_sq / ghat_n.clamp_min(1.0), n_valid]
                       + [snr_layers[k] for k in labels]).tolist()
    ce_c, ce_p, rsq, gsq, nv = vals[:5]
    snr_vals = dict(zip(labels, vals[5:]))
    return {"ce_clean": ce_c, "ce_perturbed": ce_p, "ce_gap": ce_p - ce_c,
            "reward_rms": math.sqrt(rsq / (draws_total * nv)), "ghat_rms": math.sqrt(gsq),
            "snr": (sum(snr_vals.values()) / len(snr_vals)) if snr_vals else 0.0,
            "snr_layers": snr_vals, "cosine": cosine}
