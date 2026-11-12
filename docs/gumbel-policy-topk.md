# Learned-score Top-K with Gumbel policy gradients: implementation note

`activation_bottleneck.selection_mode: learned_score_topk` with
`gumbel_policy: true`, added 2026-11-12. The method follows the Gumbel-Top-K
policy-gradient construction of Gong et al. (2026), with a hard-mask bottleneck
in place of their score-weighted aggregation. This note records what the code
does, where, and which parts are deliberate choices.

Code: `src/wsparse/bottleneck/gumbel_policy.py` (sampler, ordered
log-probability, leave-one-out loss, noise-RNG state), the `_gumbel_policy`
branch of `AdaptiveLapSumTopKGate` in `src/wsparse/bottleneck/gate.py`, the
two-branch path of `SparseTopKBottleneck` (`bottleneck/module.py`), the
learned-score branch of `TransformerLM._code_residual_stack` (`model.py`), and
the `gumbel` branch of `train()` (`train.py`). The forward-scoped collector,
`TransformerLM.forward(return_loss_details=True)` and the paired sampled
validation are shared with `laplace_policy` (`bottleneck/laplace_policy.py`).
Tests: `tests/test_gumbel_policy.py`. Configs:
`configs/bn_learned_score_gumbel.yaml` (stream placement),
`configs/cr_learned_score_gumbel.yaml` (code residual).

## 1. Architecture

Each bottleneck gets an independent linear scorer with one score per code
dimension,

$$
s = W_s h + b_s \in \mathbb R^N,
$$

where $h$ is the residual-stream state at the bottleneck. The support is
$m \in \{0,1\}^N$ with exactly $K$ ones, determined by $s$ alone; the kept
values are not scaled.

| stack | bottleneck output | scorer input $h$ |
| --- | --- | --- |
| stream (`code_residual: false`) | $W_D\,((W_E h) \odot m)$ | the bottleneck input |
| code residual, block 0 | $c_1 = (W_E h_0) \odot m$ | $h_0 = x_0 + \Delta_0(x_0)$ |
| code residual, block $l \ge 1$ | $c_{l+1} = (c_l + \alpha W_E \Delta_l) \odot m$ | $h_l = x_l + \Delta_l(x_l)$ |

Under the code residual, $x_l$ is the decoded code that block $l$ reads and
$h_l$ is the stream a stream-carried model would encode at the same site; the
carried code $c_l$ is never scored. The code-residual update rule is otherwise
unchanged.

`learned_score_topk` reuses the two-branch module of `gated_topk` (`score_proj`
next to `in_proj`, both $d_\text{model} \to N$). It requires
`surrogate_mode: hard`, and it is the only two-branch mode accepted under
`code_residual`.

**Support gradient (default, since 2026-11-12).** The binary mask passes the
scorer no gradient. With `learned_score_support_grad: true` (the default) the
scorer receives, on the support $S$ of the forward (the deterministic Top-$K$,
or the Gumbel draw),

$$
\frac{\partial L}{\partial s_i} = \mathbb 1[i \in S]\,\mathrm{sign}(v_i)\,\frac{\partial L}{\partial y_i},
$$

the signal $|a_i|$ receives under `abs_topk`, where the score is the value's
magnitude. The forward is unchanged ($y = v \odot m$; the term
$(s - \mathrm{sg}(s))\,\mathrm{sign}(v)\,m$ added to it is exactly zero);
unselected scores receive nothing. With `gumbel_policy` the policy term adds
to it, so $\lambda$ weighs the two inside the scorer.

**Gumbel over abs-TopK scores.** `gumbel_policy` also applies to
`selection_mode: abs_topk` (with `surrogate_mode: hard`): the score is $|a|$,
with $a = W_E h$ (stream), $E_0(x_0 + \Delta_0)$ at block 0 and
$c_l + E_l\Delta_l$ from block 1 under the code residual. The policy gradient
then reaches the encoders through $|a|$ and, along the carry, the earlier
blocks; $\tau$ is measured in units of the code itself.

## 2. Sampling

Training forwards (gate in `train()` mode, or `PolicyForwardSettings(sample=True)`)
draw, independently for every sequence, position and bottleneck, from a
dedicated generator:

| `gumbel_pool` | candidates $C$ | draw |
| --- | --- | --- |
| `all` (default) | all $N$ features | $(i_1, \dots, i_K) = \mathrm{TopK}_K(s/\tau + G)$ |
| `top_kj` | $C = \mathrm{TopIndices}_{K+J}(s)$, deterministic | $(i_1, \dots, i_K) = \mathrm{TopK}_K(s_C/\tau + G_C)$ |

with $G_j \sim \mathrm{Gumbel}(0, 1)$ i.i.d. $\mathrm{TopK}_K(s_C/\tau + G_C)$
orders the candidates exactly as $s_C + \tau G_C$ does. The indices are kept
in sampled order (descending perturbed score), and the mask has exactly $K$
ones. Evaluation, generation and the probes use the deterministic
$\mathrm{TopK}_K(s)$ over all $N$ features, whichever pool trains. With
`gumbel_policy: false`, training also uses the deterministic Top-$K$, and the
scorer trains through the support gradient alone.

The noise is $-\log(-\log U)$ with $U$ from `torch.rand`, clamped to
$[\text{tiny}, 1-\text{eps}]$. In float32 this truncates the lower tail at
$G \approx -2.8$ (probability $\approx 6 \times 10^{-8}$), a machine-precision
truncation.

## 3. Ordered log-probability

For the sampled order and $z = s/\tau$ (restricted to $C$ under `top_kj`):

$$
\log p(i_1, \dots, i_K \mid h) = \sum_{r=1}^{K}\Big[z_{i_r} - \log\!\!\sum_{j \in C \setminus \{i_1, \dots, i_{r-1}\}}\!\! e^{z_j}\Big].
$$

The remaining set at step $r$ consists of the never-selected candidates plus
the selected ones from rank $r$ on. The code therefore computes the
denominator as

$$
\operatorname{logaddexp}\Big(\operatorname{LSE}_{j \in C \setminus S} z_j,\;
\operatorname{LSE}_{q \ge r} z_{i_q}\Big):
$$

one masked `logsumexp` over the candidates and one reverse `logcumsumexp` over
the $K$ selected ones, in float32 with autocast disabled. Nothing is
subtracted in linear space. The result equals the direct masked-`logsumexp`
loop to $10^{-10}$ relative at score scale $10^3$ (test).

Autograd reaches every candidate's score, selected or not. The indices, the
noise and (under `top_kj`) the pool membership carry no gradient, so features
outside $C$ receive no policy gradient.

## 4. Objective, baseline and loss

Per micro-batch of $B$ sequences, the quantities are:

- $\ell_b$, the mean sampled-support CE of sequence $b$;
- $n_b$, its number of valid targets;
- $w_b = n_b / \sum_a n_a$, the weight the ordinary token-mean CE already
  gives sequence $b$ ($1/B$ for the repo's full windows);
- $P_b$, the sum of the ordered log-probabilities over the valid target
  positions of sequence $b$ and all bottlenecks.

With these:

$$
\beta_b = \frac{1}{B_\text{eff} - 1}\sum_{a \ne b} \ell_a,
\qquad
T = \lambda \sum_b w_b\, \mathrm{sg}(\ell_b - \beta_b)\, P_b,
\qquad
L_\text{backward} = L_\text{CE} + [T - \mathrm{sg}(T)].
$$

$L_\text{CE} = \sum_b w_b \ell_b$ is the existing reduction. The gradient of
$L_\text{backward}$ is that of
$\sum_b w_b[\ell_b + \lambda\, \mathrm{sg}(\ell_b - \beta_b) P_b]$. Its value
is the sampled CE, so `train/ce` keeps its meaning. One backward per
micro-batch, divided by the accumulation factor as everywhere else.

The baseline is formed inside each micro-batch, from the same sampled forward,
detached, and excluding the sequence itself. $B_\text{eff}$ counts the
sequences with at least one valid target; the others are no one's peer and
have weight 0. With $B_\text{eff} < 2$ the baseline is undefined, and the
trainer raises. The config already requires `train.micro_batch_size >= 2`.
The other sequences' supports are drawn independently of sequence $b$'s, so
$\beta_b$ does not depend on $b$'s sample.

What is exact and what is not:

1. With `gumbel_pool: all` and $\lambda = 1$, $T$ is an unbiased estimator of
   the selection part of $\nabla\, \mathbb E[L_\text{CE}]$. The ordered
   log-probability is used instead of the unordered-subset one: the loss
   depends on the set only, so the expectation marginalizes the order, and the
   cost stays at $O(N + K)$ per row. The estimator agrees with the enumerated
   gradient of a tabulated set objective within Monte Carlo error (test).
2. `top_kj` differentiates the pool-restricted policy with the pool held
   fixed. The dependence of $C$ on $s$ is piecewise constant and is not
   smoothed, as for the `laplace_policy` pool.
3. $\lambda \ne 1$ rescales the policy term only. It trades bias for variance.
   The suggested search range is $10^{-4}$ to $10^{-1}$, and the shipped
   configs start at $10^{-2}$. $\lambda = 0$ keeps the sampled forward and
   gives the scorer an exactly zero gradient.
4. $P_b$ is not normalized by $K$, by the sequence length or by the number of
   layers. The positions with ignored targets are left out of $P_b$ as
   specified. This is exact when ignored targets only trail the valid ones (a
   sampled support influences later positions only); the repo's windows have
   no ignored targets.
5. Through $h$, the policy term also reaches every parameter upstream of a
   scorer, which is part of the same gradient.

There is no activation checkpointing in the model, so no stochastic layer is
recomputed and the sampled supports of the forward are the ones the backward
uses.

## 5. Metrics

| key | meaning |
| --- | --- |
| `train/ce`, `train_stochastic/ce` | the sampled-support training CE (identical) |
| `val/ce`, `val_deterministic/ce` | deterministic $\mathrm{TopK}_K(s)$ CE; `best_val_ce` and checkpoints use it |
| `val_stochastic/ce`, `/gap`, `/mc_std` | mean over `train.policy_val_samples` Gumbel draws on the same windows (seed `train.policy_val_seed`, re-seeded per call) |
| `gumbel/tau`, `gumbel/coef` | $\tau$, $\lambda$ |
| `gumbel/advantage_rms` | RMS of $\ell_b - \beta_b$ over the window (its mean is 0 by construction) |
| `gumbel/seq_log_prob_mean` | mean $P_b$ (summed over positions and bottlenecks) |
| `gumbel/policy_term_diag` | detached $T$; a gradient diagnostic, not an objective |
| `gumbel/scorer_grad_norm` | norm of the scorers' gradient before clipping: support gradient plus policy term (0 under `abs_topk`, which has no scorer) |
| `gumbel/grad_split/*` | at `train.gumbel_grad_split_every_steps` (0: log cadence, -1: off; single process only), on the first micro-batch: `policy_norm` (the gradient of $T$ alone, one extra backward), `ce_norm` (the value gradient), `ratio`, the policy norm on the scorers and on the rest, and `ce_norm_scorer`, the support gradient on the scorers (0 with `learned_score_support_grad: false`) |
| `bottleneck/gumbel_*`, `bottleneck_gumbel_*/blocks.i` | per gate: `exchange_frac` / `overlap` against the deterministic Top-$K$, `margin` $z_{(K)} - z_{(K+1)}$ against the noise scale (std $\pi/\sqrt 6 \approx 1.28$), `logit_std`, `logp_mean`, `logp_per_select`, `pool_size`, `nonfinite` |

## 6. Random generators and checkpoints

Training noise comes from one generator per rank on the training device,
seeded `train.seed + 104729 * rank + 3`. Checkpoints carry a `gumbel_state`
payload (version 1) with every rank's generator state. A resume restores it,
and rejects a checkpoint without it or one saved with a different world size.
The baseline has no state. `load_for_inference` reads the config and weights
only.

## 7. Costs

Per gate and training micro-batch, the extra work is:

- one Gumbel draw and one `topk` over $N$ (or $K+J$) logits;
- a masked `logsumexp` and a `logcumsumexp` over $K$;
- for the backward, a float32 copy of the masked logits plus a boolean mask:
  about $5\,B T N$ bytes, or 94 MB at $B = 24$, $T = 512$, $N = 1536$
  ($5\,BT(K+J)$ under `top_kj`).

At log cadence the gradient split adds one backward of the first micro-batch.
Each sampled validation draw adds one forward per validation batch.

## 8. Rejected combinations

Under `learned_score_topk`, the config rejects:

- any `surrogate_mode` other than `hard`;
- $k \ge n_\text{features}$;
- `hard_k_annealing`, `stochastic_width` and `tie_encoder_decoder`.

The `gumbel_*` fields are rejected:

- with `gumbel_policy` off, any of them off its default;
- `gumbel_policy` with a selection mode other than `learned_score_topk` or
  `abs_topk`, or under `abs_topk` with a surrogate, `hard_k_annealing`,
  `stochastic_width` or `value_shift`;
- `learned_score_support_grad: false` outside `learned_score_topk`;
- $\tau \le 0$, $\lambda < 0$, `top_kj` with $j < 1$;
- `train.micro_batch_size < 2`.

The trainer additionally rejects:

- `learned_score_topk` without `gumbel_policy` and with
  `learned_score_support_grad: false` under DDP: the scorer has no gradient,
  and DDP runs without `find_unused_parameters`;
- a `first_order` surrogate scope.

DUST tails (`forward_from_gate`) refuse two-branch gates.

## 9. Status

Verified on CPU:

- the new tests: log-probability against the definition and normalization,
  sampling frequencies against Plackett–Luce for both pools, the unbiasedness
  test, exact-$K$ and deterministic evaluation, gradient routing, the
  leave-one-out baseline, the eval forward of both stacks against a written-out
  reference, the config rules, and training, resume and $\lambda = 0$ runs;
- the full test suite;
- `tools/repro_check.py --device cpu` for `hard`, `lapsum` and `rbk` against
  the previous commit, on synthetic tokens.

The two shipped configs also ran for 3 steps through `scripts/train_guard.py`
on CPU in bfloat16: full architecture, synthetic tokens, sequence length 128,
micro-batch 2. Every step logged exactly $K = 32$ active features and the
expected warmup LR.

Not yet run: a GPU smoke on real data. No convergence claim is made.

Measured policy-to-value gradient-norm ratios (`gumbel/grad_split/ratio`):

| model | $\lambda$ | ratio |
| --- | --- | --- |
| 2-layer CPU model | $0.1$ | 0.4–1.4 |
| shipped stream config, at initialization | $0.01$ | 1.2–3.1 |
| shipped code-residual config, at initialization | $0.01$ | 0.5–3.7 |

The smoke runs had micro-batch 2, so the baseline had a single peer (the
shipped micro-batches of 16 and 24 give it 15 and 23). These are
initialization values from 3 steps, not a calibration. They indicate that the
$\lambda$ sweep should include the lower end of $10^{-4}$ to $10^{-1}$.
