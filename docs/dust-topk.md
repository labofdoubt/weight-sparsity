# DUST for the Top-K activation bottleneck: implementation note

`activation_bottleneck.surrogate_mode: dust`, added 2026-10-22. Code:
`src/wsparse/bottleneck/dust.py` (schedule, capture, estimator, trainer
state, the per-micro-batch step), `TransformerLM.forward_from_gate` in
`src/wsparse/model.py` (the tail from a gate), the capture calls in
`SparseTopKBottleneck.forward` and `TransformerLM._code_residual_stack`, the
`dust` branch of `train()` in `src/wsparse/train.py`. Tests:
`tests/test_dust.py`. Shipped configs: `configs/bn_dust.yaml`
(`dust_target: encoder`) and `configs/bn_dust_backprop.yaml`.

The reference is "Dust" (https://qlabs.sh/research/dust): node perturbation
at every token, the population evaluated in batched forward passes, the
output error of a layer estimated from the rewards and used in the outer
product backprop would form. Here the perturbed nodes are the gate inputs of
the installed bottlenecks only, and the DUST steps alternate with ordinary
hard Top-K steps.

## 1. What is trained, and when

The gate is the hard gate: forward and backward are those of
`surrogate_mode: hard` (the tests check bit equality). `j` is the $J$ of the
Top-$(K+J)$ pool that receives noise, as it is the $J$ of RBLapSum's
candidate pool; under `dust` it never enters the gate's backward.

Optimizer steps are of two kinds, decided by the zero-based step index and
`dust_schedule = [n_hard, n_dust]`: every period of $n_{hard} + n_{dust}$
steps starts with the hard steps. `[1, 1]` (the default) alternates,
`[1, 3]` runs three DUST steps after every hard step, `[0, 1]` is pure DUST.
A hard step is the ordinary training step: CE forward, hard-mask backward,
every parameter updated. A DUST step runs no CE backward at all; what it
updates is set by `dust_target` (Section 3). Both kinds share the optimizer,
the LR schedule and `max_steps`; `train.max_steps` counts both.

## 2. The DUST step

Per micro-batch $(x, y)$ of $B$ sequences and $T$ tokens, with $L$
installed bottlenecks and $K_l = $ `dust_population` $/ L$ draws each
(`dust_population` must be a multiple of $L$ with $K_l \ge 2$; the
controller checks this at install time):

1. **Clean forward** in `train()` mode (its usage statistics are the
   training ones). Every bottleneck $l$ records its gate input
   $z_l \in \mathbb R^{B \times T \times N}$ and the input of its encoder.
   For a stream placement $z_l = W_l x_l$; under `code_residual`, for a block
   $l \ge 1$, $z_l = c_l + \alpha\, W_l \Delta_l$ is the carried code plus
   the encoded block contribution, with $\alpha = $ `code_residual_scale`.
   The clean mean CE of this forward is what the step logs as `train/ce`.
2. **Pool and noise.** For each token the clean pool
   $C_t = \operatorname{TopIndices}_{K+J}(s_t)$, $s = |z|$ under `abs_topk`,
   $s = z$ under `topk`. Draw $i$ perturbs every token at once:
   $$z'_t = z_t + \sigma_t a^{(i)}_t, \qquad a^{(i)}_t \sim \mathcal N(0, I)\ \text{on } C_t,\ 0\ \text{off it},$$
   on the signed pre-activation, before the absolute value. $\sigma_t$ is
   `dust_sigma` (`dust_sigma_mode: absolute`) or `dust_sigma` times the RMS
   of the clean pool values of the token (`relative_rms`), detached.
3. **Tail.** The forward is rerun from that gate on
   (`forward_from_gate`): decoder, post-norm, the later blocks (under
   `code_residual` the later gates add their encoded updates to the carried
   code, as in the stack), the final norm and the unembedding, the last two
   in float32. `eval()` mode under `no_grad`, so no dropout, no usage update,
   no gate diagnostics. Draws are stacked along the batch axis,
   `dust_chunk_draws` per tail forward. The per-token CE $\ell^{(i)}_s$ is
   formed in float32 (0 at ignored targets).
4. **Credit, baseline, reward.**
   $$R^{(i)}_t = \sum_{s \ge t} \gamma^{\,s - t}\, \ell^{(i)}_s, \qquad
     \tilde R_t = \frac{1}{K_l} \sum_{i=1}^{K_l} R^{(i)}_t, \qquad
     r^{(i)}_t = \tilde R_t - R^{(i)}_t,$$
   with $\gamma = $ `dust_gamma`. $\gamma = 0$ (the default) credits a
   token's noise with its own loss change only; $\gamma = 1$ with every later
   token's. The baseline is the mean over the bottleneck's own draws, as in
   the reference.
5. **Estimated output error.** The reference's Eq. 2, divided by the number
   of valid targets $N_{valid}$ so that it estimates the gradient of the mean
   token CE, the objective of the hard steps:
   $$\hat g_t = -\frac{1}{K_l\, \sigma_t\, N_{valid}} \sum_{i=1}^{K_l} r^{(i)}_t\, a^{(i)}_t
            = \frac{1}{K_l\, \sigma_t\, N_{valid}} \sum_{i} \bigl(R^{(i)}_t - \tilde R_t\bigr) a^{(i)}_t .$$
   $\hat g_t$ is nonzero on $C_t$ only. It is accumulated as
   $S_{Ra} - \tilde R\, S_a$ with $S_a = \sum_i a^{(i)}$ and
   $S_{Ra} = \sum_i R^{(i)} a^{(i)}$, so no draw is stored.

Each micro-batch of a DUST step draws its own population; with gradient
accumulation the contributions are scaled by $1 / $ `grad_accum_steps` as
the hard gradients are.

### What the estimate contains

For a loss that is differentiable at $z$, $\mathbb E[\hat g_t]$ is the
gradient of the Gaussian-smoothed loss, up to the factor $1 - 1/K_l$ that the
draw-mean baseline introduces (checked on a linear loss in the tests; the
$1/K_l$ prefactor is kept as in the reference). The Top-K selection is not
differentiable: a perturbation can move pool members in or out of the hard
support at this gate and at every later one, and that change of selection is
in the reward. This is the point of the estimate, and the reason it is
formed at the pre-activation rather than at the gate output.

## 3. `dust_target`

**`encoder`.** Only the perturbed bottleneck's encoder moves:
$$\hat G_{W_l} = \alpha_l \sum_{t} \hat g_t\, x_t^{\top}, \qquad
  \hat G_{b_l} = \alpha_l \sum_t \hat g_t,$$
with $x_t$ the encoder input of the clean forward and $\alpha_l = 1$ for a
stream bottleneck, `code_residual_scale` for a code-residual update. The
outer product is formed explicitly in float32 and added to `.grad`; no other
parameter has a gradient in that step, so the optimizer leaves it unchanged
(AdamW and the decoupled optimizer both skip parameters without a gradient;
no weight decay is applied to them either). Under `share_projections` the
one encoder receives the sum over bottlenecks.

**`dust_grad_scale`** (positive, default 1) multiplies $\hat g$ before
either target, so the whole DUST gradient scales linearly with it in both
modes while the hard steps are untouched. Section 8 gives the measured
ratio between the two gradient norms that motivates it.

**`dust_optimizer`** (`shared` | `separate`, default `shared`). Under
`shared` the run's one optimizer applies both kinds of gradient, so Adam's
first and second moments and its step count mix them; the DUST estimate's
variance then sets the second moments of every parameter it touches, and
the hard-step updates of those parameters shrink accordingly. Under
`separate` a second optimizer instance, built by the same `build_optimizer`
with the same betas, eps, weight decay and LR schedule, applies the DUST
gradient on DUST steps only, with its own moments and its own step count
(so its bias correction follows the number of DUST steps). Each kind of
update is then normalized by its own gradient statistics, and
`dust_grad_scale` has no effect on the DUST update beyond Adam's eps. Its
state is saved as `dust_optimizer` in the checkpoint and a resume requires
it. Under `model.decouple=true` the per-matrix state of `DecoupledAdamW`
splits into the parameterization (the raw gains and the sphere radius
$c_F$, which define how the fused weight is read) and the statistics (the
Adam moments and step counts of the direction and of the gains); the DUST
instance is built with `parameterization_from=` the main one, so it aliases
the main instance's parameterization tensors, which whichever instance steps
updates in place, and keeps its own statistics, the gains' moments included.
The DUST instance's checkpoint payload holds its statistics only and is
re-aliased to the main optimizer's loaded state on resume, so the main
optimizer is loaded first.

**`dust_lr_scale`** (positive, default 1) multiplies the learning rate of
the DUST steps: a DUST step uses $\eta(t) \cdot$ `dust_lr_scale` with
$\eta(t)$ the schedule's value, applied to the separate optimizer at every
step or to the shared optimizer on DUST steps only (the next step's
`set_lr` restores $\eta(t)$). The schedule and the hard steps are
untouched; `train/lr` logs $\eta(t)$, `dust/lr` the DUST steps' value.

**`backprop`.** The clean forward runs with grad enabled; after the tails,
one `torch.autograd.backward` from every $z_l$ with $\hat g_l$ as the
incoming gradient. Every parameter preceding a gate input accumulates it:
earlier blocks, earlier bottlenecks (their decoders, and their encoders
through the hard masks of the gates in between and through the carry under
`code_residual`), the embedding. The last bottleneck's decoder, the final
norm and the unembedding precede no gate input and get nothing (with tied
embeddings the embedding/unembedding matrix does get the embedding path's
gradient). The gates on the way pass the hard mask, as in a hard step.

## 4. Costs

A draw at bottleneck $l$ costs the forward of blocks $l+1 \dots L-1$ plus the
head, under `no_grad`; averaged over $l$ that is about half the block stack
plus the unembedding and CE. Measured on one A100 with `bn_dust.yaml`
(10 blocks, micro-batch 16, accumulation 4, $P = 40$): a DUST step takes
about 3.5 s against 0.9 s for a hard step, so a DUST step with population
$P$ costs roughly $1 + 0.3\,P$ forward equivalents against about $3$ for a
hard step, and the [1, 1] schedule runs at about 40% of the hard run's
throughput. The clean forward's graph is
held during the tails under `backprop` (and under `encoder` when the cosine
diagnostic is due); memory is otherwise one tail chunk's activations plus two
$[B, T, K+J]$ accumulators. Under bf16 the perturbed pre-activations are
rounded to bf16 before the gate, so `dust_sigma` should stay well above the
bf16 resolution of $z$ (about $4 \cdot 10^{-3} |z|$).

## 5. Metrics

`train/ce` is the clean hard-forward CE of the step's micro-batches on both
kinds of step, so its series is comparable with a hard run's.
`train/grad_norm` is the last step's clipped norm, whichever kind it was.
Logged per window:

- `dust/steps`: DUST steps in the window (0 on a window without one).
- `dust/ce_perturbed`, `dust/ce_gap`: the mean perturbed CE over draws,
  bottlenecks and valid tokens, and its excess over the clean CE (the price
  of the noise at this $\sigma$).
- `dust/reward_rms`: RMS of $R^{(i)}_t - \tilde R_t$ over draws and valid
  tokens; `dust/ghat_rms`: RMS of the UNSCALED $\hat g$ over the pool
  entries of valid tokens (comparable across runs with different
  `dust_grad_scale`).
- `dust/grad_norm`: mean clipped gradient norm of the window's DUST steps,
  after `dust_grad_scale`.
- `dust/cosine`, `dust_cosine/<layer>` (with `dust_cosine_every_steps` $> 0$,
  every that many DUST steps): the cosine between the encoder outer product
  of Section 3 and the hard-backward gradient of the clean CE for the same
  weight, one extra backward per micro-batch when due. Formed in both
  targets from the single-layer outer product.

Validation is unchanged: `val/ce` is the deterministic hard forward.

## 6. Random generator and checkpoints

The noise comes from a dedicated `torch.Generator` seeded from
`train.seed`, independent of the data stream, dropout and sampling.
Checkpoints carry `dust_state` (version, the generator state, the count of
completed DUST steps that paces the cosine diagnostic), and under
`dust_optimizer: separate` the second optimizer's state as `dust_optimizer`;
a resume restores them and refuses a checkpoint without them, as the policy
mode does. The step kind
is a function of the step index, so a resume continues the pattern.
`dust_chunk_draws` changes the realized noise sequence for a given seed, not
the method.

## 7. Rejected combinations

Single process only (the DUST step does not go through DDP's reducer);
bf16 or float32 (no GradScaler path); `selection_mode` `topk` or `abs_topk`;
one stream placement (`residual` or `residual_out`; the tail is not
implemented for the branch placements); `rblapsum_surrogate_scope: pool`
and every other `rblapsum_*` and `policy_*` field at its default;
`rblapsum_boundary_floor: 0`; `hard_inference: true`; no `stochastic_width`
or `value_shift` (both require `surrogate_mode: hard`). A non-default
`dust_*` value under another surrogate mode is an error.

## 8. Status

Implemented and tested on CPU (estimator closed forms, tail equivalence,
parameter movement per target, resume); both shipped configs smoke-tested for
30 steps on one A100 (georgia, 2026-10-22; `tools/repro_check.py` identical to
the previous commit for the hard, lapsum and rbk modes). No campaign has
been run; `dust_sigma`, `dust_population`, `dust_gamma` and the schedule in
the shipped configs are starting values.

What the smoke runs showed about the estimator's signal: at initialization
with $K_l = 4$ draws over a pool of $K + J = 512$, `dust/cosine` is about
$0.005$ for both targets. On a tiny model (2 blocks, $d = 64$, 256 tokens
per micro-batch) the cosine grows with the draws per bottleneck, 0.10, 0.17,
0.24, 0.31 at 4, 16, 64, 256 draws over a pool of 24, and falls with the
pool size at fixed draws, 0.14, 0.09, 0.04 at pools of 16, 64, 256 with 16
draws: roughly $\propto \sqrt{K_l / (K+J)}$ in the low-draw regime, with a
ceiling below 1 (the hard backward and the smoothed, selection-aware
estimate are different gradients). The estimate is unbiased for the
smoothed loss at any $K_l$; what $K_l$, $J$ and the number of tokens buy is
variance.

The gradient norms, measured on `bn_dust.yaml` at initialization for the
same ten encoder weights (one micro-batch of $16 \times 512$ tokens,
`dust_grad_scale: 1`):

| | hard gradient | DUST, $K_l = 4$ | $K_l = 16$ | $K_l = 64$ |
|---|---|---|---|---|
| norm over the 10 encoders | 0.112 | 82.3 | 46.0 | 23.6 |
| ratio to the hard gradient | 1 | 735 | 410 | 210 |
| cosine with the hard gradient | | 0.005 | 0.009 | 0.020 |

The DUST norm falls as $1/\sqrt{K_l}$: it is set by the estimator's
variance, while the signal component ($\cos \times$ norm, 0.4 to 0.5) is
within a factor of 4 of the hard gradient. The per-encoder ratio runs from
about 100 for the last block to about 50 000 for the first, because at
initialization the hard gradient of the early encoders is tiny while the
noise norm is nearly the same in every block. With `grad_clip: 1.0` and
`dust_grad_scale: 1` every DUST step is clipped by a factor 20 to 80 and no
hard step is, and Adam's second moments for the encoders are set by the
DUST steps. `dust_grad_scale` is the knob for balancing the two kinds; the
shipped configs keep it at 1.
