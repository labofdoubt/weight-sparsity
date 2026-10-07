# Laplace-policy hard Top-K: implementation note

`activation_bottleneck.surrogate_mode: laplace_policy`, added 2026-10-08.
The specification this implements is `docs/laplace-policy-topk-implementation.md`;
this note records what the code does, where, and which parts are deliberate
approximations. Code: `src/wsparse/bottleneck/laplace_policy.py` (sampler,
density, schedule, records, loss, trainer state), the `_laplace_policy` branch
of `AdaptiveLapSumTopKGate` in `src/wsparse/bottleneck/gate.py`,
`TransformerLM.forward(return_loss_details=True)` in `src/wsparse/model.py`,
the trainer branch in `src/wsparse/train.py`. Tests: `tests/test_laplace_policy.py`,
plus the guard and (opt-in) DDP cases in `tests/test_train_ddp.py`.

## 1. Forward

For one gate row with signed encoder output $z \in \mathbb R^N$ and clean
scores $s$ ($|z|$ under `abs_topk`, $z$ under `topk`):

1. $C = \operatorname{TopIndices}_{K+J}(s)$ from the clean scores. The indices
   are discrete; the gathered scores keep their autograd path.
2. $u_i = s_i - \frac{1}{K+J}\sum_{j \in C} s_j$ with `policy_center_scores`
   (differentiable mean), else $u_i = s_i$. The width $T$ is computed from
   the uncentred clean scores.
3. $r_i = u_i + \varepsilon_i$, $\varepsilon_i \sim \mathrm{Laplace}(0, T)$
   independent per row, candidate, gate call and micro-batch, from a
   dedicated generator. $r$ is detached.
4. The $K$ largest $r_i$ form the support; their original $z_i$ are
   scattered back, everything else is 0. Exactly $K$ indices, all inside $C$.

Sampling happens iff the gate is in `train()` mode, unless a per-forward
`PolicyForwardSettings(sample=...)` says otherwise. It never depends on
`torch.is_grad_enabled()`. Evaluation, `generate()` and the probes use the
clean deterministic Top-K. With $J = 0$ or $K = N$ no exchange is possible: the
gate is the ordinary hard gate with no noise and no density term.

## 2. Objective and estimator

The reference objective is the expected sampled CE,
$\mathcal J_T(\theta) = \mathbb E_{r \sim \rho_{\theta, T}}[L_{\mathrm{CE}}(\theta; r)]$.
Per micro-batch, with $\ell_{bt}$ the unreduced CE (ignored targets at 0),
$n_b$ the valid targets of sequence $b$, $N_{\mathrm{valid}} = \sum_b n_b$,
$c_b = \frac{1}{n_b}\sum_t \ell_{bt}$ and $w_b = n_b / N_{\mathrm{valid}}$:

$$
L_{\mathrm{CE}} = \sum_b w_b c_b, \qquad
L_{\mathrm{support}} = \sum_b w_b\, \mathrm{sg}(c_b - B) \sum_{a \in b} \gamma_a
\log \rho_{T_a}\!\left(\mathrm{sg}(r_a) \mid u_a\right),
$$

$$
\log \rho_T(r \mid u) = \sum_{i \in C}\left[-\log(2T) - \frac{|r_i - u_i|}{T}\right],
\qquad
L_{\mathrm{backward}} = L_{\mathrm{CE}} + \left[L_{\mathrm{support}} - \mathrm{sg}(L_{\mathrm{support}})\right].
$$

The sum over $a$ runs over every sampled gate instance (all layers,
placements, positions) of the sequence; nothing is averaged over $K+J$,
sequence length or layers. The numerical value of the loss is the sampled CE;
its gradient is the hard-mask value gradient plus the selection gradient. The
trainer divides the whole computational loss by the accumulation factor and
backpropagates once per micro-batch.

Detach rules: the complete sampled $r$, the width $T$, the advantage
$c_b - B$ and $\gamma$ are detached; $u$ and its centring mean are not. For
uncentred scores the row location gradient is $w_b \gamma_a (c_b - B)\,
\mathrm{sign}(r_i - u_i) / T_a$; `abs_topk` multiplies by $\mathrm{sign}(z_i)$.

What is exact and what is not:

1. With fixed pool membership, an externally set absolute $T$, a
   sample-independent baseline and $\gamma = 1$ the estimator is unbiased for
   $\nabla \mathcal J_T$ at that step's temperature (verified against the
   analytic two-candidate derivative and an enumerated two-gate objective).
2. Pool changes are not smoothed: the construction differentiates selection
   within the current pool and the smooth computations on rank-stable regions.
3. The relative widths use $T = \tau\, a(\theta)$ with $a$ detached: a
   frozen-scale partial gradient. The derivative through the
   activation-dependent width is not implemented and there is no
   `differentiate_temperature` option.
4. $\gamma \ne 1$ rescales the selection gradient only. It is a stabilization
   or ablation, not the gradient of the expected CE.

A later gate's density term backpropagates into earlier blocks through the
sampled hard values (tested: detaching that path fails the enumerated
reference); no surrogate Jacobians are multiplied along a code-residual carry
and no `first_order` scope is involved.

## 3. Temperature, schedule, rescaling

Per row, from the clean uncentred sorted pool ($s_{(K+1)}$ is index $K$,
$s_{(K+J)}$ index $K+J-1$):

| `policy_temperature_mode` | $T$ | requires |
| --- | --- | --- |
| `absolute` | $\max(T_{\min}, \tau)$ | |
| `relative_b` | $\max(T_{\min}, \tau\, s_{(K+1)})$ | `abs_topk` |
| `relative_span` | $\max(T_{\min}, \tau\,[s_{(K+1)} - s_{(K+J)}])$ | $J \ge 2$ |

$T_{\min}$ = `policy_min_temperature` (default `1e-6`) is a numerical floor,
not an annealing endpoint; `bottleneck/policy_t_floor_frac` logs how often it
binds and `policy_span_zero_frac` how often the span is zero.

Schedule, over the zero-based step $t$ of the LR schedule, with
$h$ = `policy_temperature_hold_steps`, $d$ = `policy_temperature_anneal_steps`:
$v(t) = \mathrm{clip}((t - h)/d, 0, 1)$,
$\tau(t) = \tau_0 (\tau_f / \tau_0)^{v(t)}$ (`exponential`), or $\tau_0$
(`constant`). $\tau_0$ is `activation_bottleneck.temperature`, $\tau_f$ =
`policy_temperature_final` $\le \tau_0$. The controller's `set_step` writes
$\tau(t)$ to every gate's `policy_tau` before the `on_step` probe and the
micro-batches; the configured value is never overwritten, so the dumped config
keeps the schedule and a resume recomputes $\tau$ from the restored step.

Support multiplier $\gamma_a$ (`policy_support_scale` = $\gamma_0$, reference
`policy_support_temperature_ref`, cap `policy_support_scale_max`):

| `policy_support_scale_mode` | $\gamma_a$ | effect |
| --- | --- | --- |
| `constant` | $\gamma_0$ | reference ($\gamma_0 = 1$); $0$ trains values only |
| `effective_temperature` | $\gamma_0 T_a / T_{\mathrm{ref}}$ | cancels the sampled $1/T_a$ |
| `scheduled_temperature` | $\gamma_0 \tau(t) / \tau_{\mathrm{ref}}$ | cancels the annealed $1/\tau$, keeps the inverse row scale |

$\gamma$ multiplies each row's log density before the sequence reduction;
the CE, the value path, the LR and the optimizer are never scaled.

## 4. Baseline

`policy_baseline: ema` (default) keeps a scalar $B$ in mean-CE units,
initialized at `policy_baseline_initial` (null: $\log V$), updated once per
optimizer step as $B \leftarrow \delta B + (1 - \delta)\,\bar c$ with
$\bar c$ the step's token-weighted mean CE over all micro-batches and (under
DDP, by an all-reduce on every rank) all ranks. $B$ is held fixed within a
step; validation and probes never touch it. `none` sets $B = 0$. The
same-batch mean is deliberately not used as a baseline.

## 5. Evaluation and metrics

| key | meaning |
| --- | --- |
| `train/ce`, `train_stochastic/ce` | the sampled-support training CE (identical) |
| `train_deterministic_probe/ce`, `/paired_stochastic_ce`, `/gap` | noise-off `eval()`/`no_grad` forward on the step's own inputs before the update, at `policy_train_deterministic_every_steps` (0: `log_every_steps`, -1: off); one extra forward per probed micro-batch; dropout is off in the probe, so with nonzero dropout it is not a controlled pair |
| `val/ce`, `val_deterministic/ce` | clean deterministic Top-K CE (identical); `best_val_ce` and checkpoints use it |
| `val_stochastic/ce`, `/ppl`, `/samples`, `/mc_std`, `/gap` | mean over `policy_val_samples` draws on the same windows, dropout off, at the current $\tau$; `mc_std` is the sample std of the draw means and only exists for two or more draws; each draw costs one forward per validation batch |
| `val_final_*` | the same keys at the final validation |
| `policy/tau`, `policy/baseline`, `policy/advantage_mean`, `policy/advantage_rms` | the step's schedule value, the baseline used, the detached advantages |
| `policy/support_term_diag` | detached value of $L_{\mathrm{support}}$: a gradient diagnostic, not an objective, never a perplexity |
| `bottleneck/policy_*`, `bottleneck_policy_*/blocks.i` | per-gate (and block-averaged) statistics of what was sampled: effective $T$ mean/min/max, floor fraction, $\gamma$, exchange fraction and overlap with the clean Top-K, pool span, log density, collapsed perturbations ($r = u$), RMS of the raw and scaled location gradient, zero-sum residual, a non-finite flag |

The stochastic validation and the probe run on the unwrapped model with a
dedicated evaluation generator re-seeded from `policy_val_seed` at every call,
so they are reproducible across checkpoints and leave the training RNG, the
baseline, the schedule and the usage statistics untouched.

`scripts/train_guard.py` aborts on a non-finite value of any of the CE keys
above; its thresholds still apply to `train/ce`.

## 6. Random generators and checkpoints

Training noise comes from one `torch.Generator` per rank on the training
device, seeded `train.seed + 104729 * rank + 1`; a backend without a device
generator raises instead of falling back to the global RNG. Checkpoints carry
an optional `policy_state` payload (`version` 1): baseline mode and value,
update count, decay, world size and every rank's generator state (gathered
with a collective on all ranks before the rank-0 save). A `laplace_policy`
resume restores it and rejects a checkpoint without it or with a different
world size; starting from another run's weights is a fresh training.
`load_for_inference` reads config and weights only and returns the model in
`eval()`.

Guarantee: on fixed supplied inputs, restoring the state reproduces the
subsequent supports and computational gradients (tested). This is not a
bit-for-bit whole-run resumption: the data stream RNG and the global
(dropout) RNG are not checkpointed by this trainer, as before.

## 7. Costs

Per training micro-batch: one `topk` over $K+J$ noisy scores per gate row,
$K+J$ Laplace variates and the density in float32, and one extra noise-off
forward per micro-batch at the probe cadence. Per validation: `policy_val_samples`
extra forwards per batch. Per-gate diagnostics add scalars at log cadence
only.

## 8. Rejected combinations

`gated_topk`; `stochastic_width`; `value_shift`; a positive
`rblapsum_boundary_floor`; any `rblapsum_*` knob or scope off its default;
`hard_inference: false`; `relative_span` with $J < 2$; `relative_b` without
`abs_topk`; non-finite or non-positive widths, decays outside $[0, 1)$,
`policy_temperature_final` above $\tau_0$ or missing for `exponential`, a
nonzero hold/anneal or an endpoint under `constant`; any `policy_*` field off
its default under another surrogate mode. The removed legacy schedule names
(`temperature_schedule`, `temperature_start`, ..., `differentiate_temperature`,
`reinforce_*`) are still dropped by the migration and never map onto the
`policy_*` fields.

## 9. Status

Verified on CPU: the test suite, the estimator tests, and
`tools/repro_check.py --device cpu` for `hard`, `lapsum` and `rbk` against the
previous commit. Not yet run: a GPU smoke of the shipped configs and the
opt-in two-process DDP test (`WSPARSE_DDP_TEST=1`), which need a box. A short
smoke is a plumbing check, not evidence of convergence; no convergence claim
is made for this method.
