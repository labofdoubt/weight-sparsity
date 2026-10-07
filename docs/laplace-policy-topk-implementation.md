# Implementation handoff: Laplace-policy hard Top-K

## 1. Task and scope

Implement a new activation-bottleneck training method in this repository. Use
`activation_bottleneck.surrogate_mode: laplace_policy` as its canonical name.
This is a new mode, not a change to the numerical behavior of `rblapsum`,
`rblapsum_sf`, `lapsum`, `soft_ste`, or `hard`.

The method samples the membership of an exactly K-element support from the
current clean Top-(K+J) pool, transmits unmodified encoder values, and trains
selection using a likelihood-ratio estimator of actual stochastic CE. All
bottlenecks are trained together in one stochastic forward/backward per
microbatch. No explicit probability distribution over subsets is required.

Required deliverables:

- A complete new training mode, integrated into the existing model, trainer,
  configuration dataclasses, YAML/CLI loading, evaluation, and checkpoints.
- Stochastic and deterministic CE metrics, including paired validation and
  periodic deterministic training probes.
- Constant and exponentially annealed temperature, in absolute and relative
  scaling. Relative temperature must be detached: do not differentiate through
  its activation-dependent scale in this implementation.
- Optional rescaling of the support-selection gradient only, including a
  temperature-proportional option.
- Tests of the estimator itself, multi-bottleneck propagation, numerical
  stability, metrics, configuration round trips, and checkpoint state.
- Shipped examples and documentation. Preserve old modes and archived-config
  reconstruction; do not silently reinterpret an old experiment.

Read `AGENTS.md` and relevant parts of `docs/vastai-agent-guide.md` before
working. Code and notes are edited locally; real GPU smoke runs require a
current box supplied by the user. Do not infer a live box from old notes.
Keep `train.compile=False`.

This handoff was prepared against commit
`feb74a88717c5e380c933df6c6c1eda3b8cbc179`. Recheck the implementation if the
checkout has moved. The handoff is a specification, not implemented code.

## 2. Forward definition

### 2.1 Values, scores, and the candidate pool

For one gate row, let z be the signed encoder output, N its feature count,
K the support size, and J the number of additional candidates. For
`selection_mode=abs_topk`, scores are s_i = |z_i|. For `topk`, scores are
s_i = z_i. Initially support these two selection modes; reject `gated_topk`
for the new mode rather than assuming that its existing constrained-LapSum
backward is applicable.

Compute the candidate indices from the clean scores, before adding noise:

$$
C=\operatorname{TopIndices}_{K+J}(\boldsymbol{s}).
$$

Gather both the scores and original values on C. Candidate indices are
discrete and fixed during backward, but the gathered scores must retain their
autograd connection. Do not copy the existing RBLapSum convention that
detaches `cand_scores` for kernel geometry: that would eliminate policy
learning here.

The pool is recomputed on every forward. Noise may select any K members of
this pool, including dropping originally active candidates. It must not
introduce candidates outside this clean pool, enlarge the support beyond K,
or merely sample a prefix length. The existing `stochastic_width` feature is
a different method and must not be combined with this one.

### 2.2 Optional score centering

With `policy_center_scores=true`, use

$$
u_i=s_i-\frac{1}{K+J}\sum_{j\in C}s_j,\qquad i\in C.
$$

Otherwise use u_i = s_i. Compute the mean with gradients enabled: do not
detach it. Center only within each row's candidate pool, not across tokens,
sequences, or layers. Compute temperature geometry from the uncentered clean
scores.

For a fixed row width, centering does not change the support distribution:
subtracting a common constant leaves noisy-score rankings unchanged. It
projects the sampled policy score gradient onto the zero-sum subspace,
removing its irrelevant uniform-score-shift component. See Section 5.

### 2.3 Sample scores and keep exactly K original values

For effective temperature T > 0, sample independent Laplace noise:

$$
\varepsilon_i\sim\operatorname{Laplace}(0,T),\qquad
r_i=u_i+\varepsilon_i,\qquad i\in C.
$$

Choose the K largest r_i, and scatter their original signed values z_i back
to the full N-feature row. The output is

$$
y_i=z_i m_i,\qquad
m_i=\boldsymbol{1}\{i\in\operatorname{TopIndices}_K(\boldsymbol r_C)\}.
$$

Every stochastic forward is already a hard forward. Temperature controls
randomness of membership, not softness of transmitted values. Do not apply
an activation threshold after noisy Top-K, add noise to values, multiply
values by probabilities, or use a straight-through mask.

Exactly K means K selected indices. If a selected z_i is exactly zero, the
number of nonzero output values may be smaller. Diagnose cardinality using
the mask/indices, not `y != 0`.

Default evaluation and generation use clean deterministic Top-K with no
noise. Stochastic evaluation explicitly overrides sampling while keeping
the model in `eval()` mode. Sampling must not depend on whether gradients
are enabled: stochastic validation runs under `no_grad()`.

## 3. Objective and backward estimator

### 3.1 The reference objective

For fixed absolute temperature, the reference objective is the expected
actual CE of the randomized hard-support network:

$$
\mathcal J_T(\theta)
=\mathbb E_{\boldsymbol r\sim\rho_{\theta,T}}
\left[L_{\mathrm{CE}}(\theta;\boldsymbol r)\right].
$$

Top-K enters inside the network defining this CE. The conditional density
of one row's noisy scores is available directly:

$$
\log\rho_T(\boldsymbol r\mid\boldsymbol u)
=\sum_{i\in C}
\left[-\log(2T)-\frac{|r_i-u_i|}{T}\right].
$$

With many bottlenecks the joint density factors into these conditional row
densities. Later scores depend on earlier sampled supports and on network
parameters. Do not treat later gates as independent of earlier activations.

The standard likelihood-ratio identity supplies an ordinary sampled-value
gradient and a selection gradient. Its generic stochastic-computation-graph
justification is described in
[Schulman et al., 2015](https://papers.nips.cc/paper_files/paper/2015/file/de03beffeed9da5f3639a621bcab5dd4-Paper.pdf).
The configuration-specific qualifications are in Section 3.5.

For one sequence and gamma=1, the identity to implement is

$$
\nabla_\theta\mathcal J_T
=\mathbb E\left[
\left.\nabla_\theta c_b\right|_{\boldsymbol r\ \mathrm{fixed}}
+(c_b-B)\sum_{a\in b}
\left.\nabla_\theta\log\rho_{T_a}(\boldsymbol r_a\mid\boldsymbol u_a)
\right|_{\boldsymbol r\ \mathrm{fixed}}
\right].
$$

The ordinary CE derivative holds all sampled supports fixed. The density
derivatives hold all sampled noisy-score vectors fixed, while differentiating
their conditional means through the intervening sampled-value computations.

### 3.2 Loss reduction and credit assignment

Use the CE of the whole sequence as the default cost for each gate row in
that sequence. Gates can affect later token predictions through causal
attention. Multiplying a row's policy term only by CE at the same token is
generally incorrect. Do not detach the graph connecting later gate scores
to earlier deterministic computations.

For a microbatch, let ell_bt be unreduced CE, with ignored targets masked
out. Define n_b as the number of valid targets in sequence b, N_valid as
their total, c_b as each nonempty sequence's mean CE, and w_b = n_b/N_valid.
The ordinary loss is L_CE = sum_b w_b c_b. Sequences with no valid targets
have zero weight; an entirely ignored microbatch must fail clearly rather
than produce NaN.

Let a index gate-row instances in sequence b, including all layers,
placements, and positions. Let B be a detached baseline in mean-CE units,
and gamma_a a detached optional support multiplier. Construct

$$
L_{\mathrm{support}}
=\sum_b w_b\,\operatorname{stopgrad}(c_b-B)
\sum_{a\in b}\gamma_a
\log\rho_{T_a}\!\left(
\operatorname{stopgrad}(\boldsymbol r_a)\mid\boldsymbol u_a
\right).
$$

The computational loss can be written as

$$
L_{\mathrm{backward}}
=L_{\mathrm{CE}}
+\left[L_{\mathrm{support}}
-\operatorname{stopgrad}(L_{\mathrm{support}})\right].
$$

The zero-valued bracket leaves the numerical scalar equal to sampled CE
while adding the required gradient. It is not a second supervised loss to
report as CE. Log the actual losses separately from gradient diagnostics.

Sum log densities over candidates and gate instances. Do not average them
over K+J, sequence length, or number of layers. Such averaging changes the
relative strength of the selection estimator. Only the CE/data reduction
and the explicitly configured gamma should set its scale.

Apply the trainer's gradient-accumulation factor to the entire computational
loss, not only CE. Match the existing per-microbatch and DDP loss reduction;
do not silently change it to a different global token-weighted objective.
For this repo's usual equal-length unmasked batches these reductions agree.

### 3.3 Essential detach rules and gradient sign

- Detach the complete sampled r in `log_prob`. Detaching only epsilon is
  insufficient: differentiating r = u + epsilon in the density cancels the
  dependence on u and makes the location gradient zero.
- Keep u differentiable, including its optional candidate-pool mean.
- Detach c_b - B, T, and gamma in the support term. A coefficient whose
  gradient leaks through CE creates an unintended second CE path.
- Do not use `rsample()` and then ordinary backpropagation through Top-K as
  a substitute for the likelihood-ratio term. Hard index selection loses
  the membership gradient.
- Do not reverse the sign as if CE were a reward. This is loss minimization:
  the coefficient is +(CE - baseline) times log density.

For uncentered scores and frozen T, the raw row location gradient is

$$
\frac{\partial L_{\mathrm{support}}}{\partial s_i}
=w_b\gamma_a(c_b-B)
\frac{\operatorname{sign}(r_i-u_i)}{T_a}.
$$

For abs-topk the score path additionally multiplies by sign(z_i). The
ordinary value path remains exactly the sampled hard-mask CE gradient.
Use ordinary autograd to combine them, not the existing custom RBLapSum VJP.

### 3.4 Minimal computation outline

The following is pseudocode for one row plus the sequence-level loss, not
drop-in repository code. It makes the distinct detach rules explicit:

```python
# Existing forward-computed clean scores; candidate IDs have no derivative.
score_c, candidate_ids = topk(clean_scores.float(), K + J, sorted=True)
value_c = gather(original_signed_values, candidate_ids)
T = effective_width(score_c.detach(), scheduled_tau).detach()
# T has shape score_c.shape[:-1] + (1,), including in absolute mode.
u = score_c - score_c.mean(-1, keepdim=True) if center_scores else score_c

# Device-specific independent noise. Do not retain a path through the sample.
epsilon = sample_laplace(T, shape=u.shape, generator=training_noise_generator)
r = (u.detach() + epsilon).detach()
selected_local_ids = topk(r, K).indices
mask_c = scatter_binary_mask(selected_local_ids, pool_size=K + J)
output = scatter_to_full_row(value_c * mask_c, candidate_ids)

# Keep u differentiable. Density includes ALL sampled pool coordinates.
log_prob_row = (-log(2 * T) - abs(r - u) / T).sum(-1)
gamma_row = support_multiplier(T, scheduled_tau).detach().squeeze(-1)
collector.add(gamma_row * log_prob_row)

# Later, after unreduced CE and all rows in the sequence are available:
log_prob_sequence = sum_collected_rows_by_sequence()
advantage = (sequence_mean_ce - previous_step_baseline).detach()
support = (valid_token_weights * advantage * log_prob_sequence).sum()
loss_for_backward = ordinary_ce + (support - support.detach())
```

The production branch must also bypass density when sampling is disabled
or no exchange is possible, manage collector lifetime, and handle shape,
precision, generator, and evaluation requirements described below.

### 3.5 What is exact and what is deliberately modified

Keep these distinctions in code comments and documentation:

1. With fixed candidate membership, externally specified absolute T, valid
   baselines, and gamma=1, the estimator is unbiased for the reference
   stochastic CE objective. Temperature may change externally between
   optimizer steps; the objective at each step uses that step's temperature.
2. A deterministic Top-(K+J) candidate pool has rank-membership
   discontinuities. The construction differentiates selection within the
   current pool and the smooth computations on rank-stable regions. It does
   not smooth candidate-pool changes or optimize the unrestricted full-N
   noisy-selection objective. This restriction is intentional.
3. Relative scaling with T = tau * a(theta) but detached a is a frozen-scale
   partial-gradient rule. It omits the scale derivative of the full
   activation-dependent-noise objective. This is explicitly requested for
   version 1; do not describe this configuration as an exact full gradient.
4. Gamma different from one rescales only part of the gradient. It is an
   intentional stabilization/ablation, generally not the gradient of the
   original expected CE objective. Do not hide it as a variance-only change.

These qualifications do not justify arbitrary truncation between layers.
The reference estimator's later log-density terms must backpropagate into
earlier blocks through the ordinary sampled hard-value computation. The
earlier gate's own likelihood-ratio term separately accounts for changing
that earlier support. There is no multiplication of hand-written surrogate
Jacobians across gates, and no need for an RBLapSum `first_order` scope here.

## 4. Temperature and support rescaling

### 4.1 Temperature modes

Reuse `activation_bottleneck.temperature` as the initial scheduled parameter
tau_0, without changing its meaning for old modes. Add separate policy
fields; never reuse migration-stripped legacy schedule names.

The new modes are:

- `absolute`: T_row = max(policy_min_temperature, tau(step)). Tau has score
  units.
- `relative_b`: T_row = max(policy_min_temperature, tau(step) * s_(K+1)).
  Tau is dimensionless. Initially allow this only with abs-topk so the rank
  boundary is nonnegative.
- `relative_span`: T_row = max(policy_min_temperature,
  tau(step) * [s_(K+1) - s_(K+J)]). Tau is dimensionless. This uses the same
  inactive-pool span as the current implementation, not s_(K) - s_(K+J).

Use clean scores from the current row, not noisy scores or centered scores.
With zero-based sorted candidate indexing, these ranks are candidate indices
K and K+J-1. Detach the whole effective width in both relative modes.
The absolute minimum width is a numerical safeguard, not the annealing
endpoint. Log how often it binds. A frequently binding floor invalidates
the interpretation that relative noise remains scale-proportional.

`relative_span` needs J >= 2 when exchanges are possible: for J=1 its span
is identically zero. Reject that setting explicitly. Handle zero measured
spans through the positive width floor, and report them.

### 4.2 Schedule

Implement `constant` and `exponential`. For exponential, use an initial
hold of h optimizer-loop steps and an anneal duration d > 0:

$$
v(t)=\operatorname{clip}\left(\frac{t-h}{d},0,1\right),\qquad
\tau(t)=\tau_0\left(\frac{\tau_f}{\tau_0}\right)^{v(t)}.
$$

Here t is the same zero-based training-step index used by the LR schedule.
Hold tau_f after the anneal. Both endpoints must be positive; an annealing
configuration must have tau_f <= tau_0. Constant mode keeps tau_0.

In absolute mode anneal the absolute width parameter. In relative modes
anneal the dimensionless multiplier, then recompute each row's detached
activation scale. Do not try to hold an absolute T trajectory in relative
mode. Use the same schedule value throughout all accumulation microbatches
and all gates in an optimizer step. Validation/probes must not advance it.

Do not set T=0 and then evaluate a density. Hard deterministic inference
already uses a separate noise-off path. A positive final width is required
for stochastic training/evaluation. A noise-off fine-tuning phase is outside
this initial implementation.

There is no established optimal annealing schedule for this exact method.
The example settings below are starting configurations, not a convergence
claim. Annealing changes the stochastic objective and the train/inference
gap; it does not guarantee deterministic CE improvement.

### 4.3 Optional support-only rescaling

Add a nonnegative gamma_0 and three modes. All multipliers are detached.

$$
\gamma_a=
\begin{cases}
\gamma_0,&\texttt{constant},\\
\gamma_0 T_a/T_{\mathrm{ref}},&\texttt{effective\_temperature},\\
\gamma_0\tau(t)/\tau_{\mathrm{ref}},&\texttt{scheduled\_temperature}.
\end{cases}
$$

Use one positive configured reference parameter; its units follow the
selected mode. In `constant` mode that reference is unused. An optional
maximum can cap gamma from above. Gamma=0 is a supported value-gradient-only
ablation; it does not itself disable forward support sampling.

`effective_temperature` cancels the explicit 1/T factor of the sampled row
score gradient. `scheduled_temperature` cancels the annealed 1/tau factor
but retains the inverse row scale in relative mode. Holding the cost
coefficient fixed, the latter preserves the inverse-score-scale
transformation of the density location derivative away from numerical
floors; effective-width scaling does not have that same property. This is
not a claim that the complete network CE is scale invariant. Offer both
and state the distinction, rather than using an ambiguous flag named `scale_by_T`.

Apply row-dependent gamma before reducing row log densities into sequence
totals. Do not replace it with a batch-mean T. Do not scale the entire CE,
the hard value path, learning rate, or optimizer gradient by gamma.

This controls an explicit source of amplification, not all possible
divergence. The score-gradient second moment can grow strongly as T shrinks.
The expected gradient is not universally proportional to 1/T: away from
support-exchange boundaries it may instead approach zero. Keep ordinary
gradient clipping, a positive width floor, and useful diagnostics.

## 5. Baseline, variance, and zero-sum selection gradients

Implement `policy_baseline: none | ema` initially. A learned critic,
counterfactual forwards, and a deterministic-forward baseline are not
required in version 1.

For `none`, B=0. For `ema`, maintain a previous-step scalar mean-CE baseline.
Initialize it from `policy_baseline_initial`, or log(vocab_size) when the
field is null. This initialization does not depend on the current sampled
supports. Use a configured EMA decay. Hold B fixed throughout all
microbatches of a step and update only after their advantages have been
formed/backpropagated. Update from detached CE statistics, not the
computational support term. Under DDP, all-reduce the CE sum and valid-token
count on all ranks so every rank uses the same baseline on the next step.

Do not subtract the current minibatch mean containing the same sample as
a purported unbiased baseline. Detaching that mean does not remove its
statistical dependence on the sampled support. Likewise do not normalize
advantages by their same-batch standard deviation in the reference mode.
Additional biased stabilization may be studied later, but must be separate.

High variance is a central risk: there are many row/candidate decisions and
each may have a small effect on sequence CE. The full noisy-score density
contains randomness that changes no support at all. An unbiased estimator
is not necessarily a practical low-variance estimator. If the basic method
is poor, report that result before adding an undocumented RBLapSum term.

Exactly-K inclusion probabilities mu_i satisfy sum_i mu_i = K. However,
raw sample gradients of noisy-score log density do not sum to zero. Do not
reuse `through_rank_kappa` or claim that constant cardinality makes the raw
estimator zero-sum sample by sample.

With frozen T and fixed pool, optional centering in Section 2.2 yields

$$
h_i'=h_i-\frac{1}{K+J}\sum_{j\in C}h_j,
\qquad \sum_{i\in C}h_i'=0.
$$

This is the correct uniform-score-shift projection for the frozen-width
support distribution. It does not change its expected policy gradient. It
is not the kernel-weighted correction used by current RBLapSum. In relative
mode, describe this property for the frozen-width score derivative, not for
the omitted full temperature derivative. Test the zero sum in score space;
after abs-topk's sign(z) chain rule, gradients in z-space need not sum to zero.

## 6. Configuration contract

Add these fields to `ActivationBottleneckConfig` in `src/wsparse/config.py`.
Use the same names in YAML, CLI overrides, saved configs, and checkpoints.
Validation must reject unsupported combinations before model execution.

| Field | Default | Meaning |
| --- | --- | --- |
| `surrogate_mode` | existing default unchanged | Add `laplace_policy` to the allowed modes. |
| `temperature` | existing default unchanged | Initial tau for this mode. |
| `policy_temperature_mode` | `absolute` | `absolute`, `relative_b`, or `relative_span`. |
| `policy_temperature_schedule` | `constant` | `constant` or `exponential`. |
| `policy_temperature_final` | `null` | Required positive endpoint for exponential. |
| `policy_temperature_hold_steps` | `0` | Initial hold, nonnegative. |
| `policy_temperature_anneal_steps` | `0` | Positive duration for exponential. |
| `policy_min_temperature` | `1e-6` | Positive effective-width floor in score units. |
| `policy_center_scores` | `true` | Per-row candidate-score centering with differentiable mean. |
| `policy_baseline` | `ema` | `none` or `ema`. |
| `policy_baseline_decay` | `0.99` | EMA decay in [0,1). |
| `policy_baseline_initial` | `null` | Optional finite initial mean CE; null means log(vocab_size). |
| `policy_support_scale` | `1.0` | Nonnegative gamma_0; 1 is the reference scale. |
| `policy_support_scale_mode` | `constant` | `constant`, `effective_temperature`, or `scheduled_temperature`. |
| `policy_support_temperature_ref` | `1.0` | Positive reference width/multiplier for rescaling. |
| `policy_support_scale_max` | `null` | Optional positive upper cap on gamma. |

Detached relative width is mandatory in this version, not a silently
ignored configurable option. Document it explicitly. Do not expose a
`differentiate_temperature=true` setting unless the full derivative and its
tests are actually implemented later.

Add these fields to `TrainConfig`; they only affect the new mode:

| Field | Default | Meaning |
| --- | --- | --- |
| `policy_val_samples` | `1` | Positive number of stochastic support draws per validation batch. |
| `policy_val_seed` | `1337` | Dedicated reproducible evaluation seed, independent of training RNG. |
| `policy_train_deterministic_every_steps` | `0` | 0 inherits `log_every_steps`; positive sets cadence; -1 explicitly disables the extra training probe. |

For the new mode, the default therefore tracks deterministic training CE
periodically, not on every microbatch. Explain the extra-forward cost when
this probe is enabled. Always retain both validation losses, regardless of
whether the optional training probe is disabled.

Version-1 validation rules:

- 1 <= K <= N, J >= 0, K+J <= N. If J=0, or K=N, selection is fixed; bypass
  noise and policy density entirely and use the ordinary hard-value path.
  Do not inject meaningless density gradients when there is no exchange.
- Require J>=2 for nontrivial `relative_span`, and abs-topk for nontrivial
  `relative_b`.
- Reject `gated_topk`, non-`none` `stochastic_width`, and non-`none`
  `value_shift` for this mode initially.
- Do not use a positive RBLapSum boundary floor to delete selected features.
  Reject a nonzero `rblapsum_boundary_floor` under the new mode rather than
  importing its capped-support semantics.
- Require the existing scope to be its neutral/default `pool`; reject
  `first_order`, `update*`, `carry*`, inactive-only, or other RBLapSum routing.
- Nondefault RBLapSum support strengths, permutations, relative-width
  switches, token centering, and SF-specific settings must not silently
  alter this method. Reject incompatible overrides or clearly report them
  as invalid; leave harmless existing defaults alone.
- Validate all floats for finiteness, not just positivity: NaN can pass a
  naive `value <= 0` check. Validate integer cadence/duration bounds.
- Keep old defaults and migration behavior for old modes. Unsupported
  policy-field overrides under other modes should fail clearly rather
  than suggest that annealing was applied to old RBLapSum.

Important: `_migrate_legacy()` currently discards names including
`temperature_schedule`, `temperature_start`, `temperature_end`,
`temperature_anneal_steps`, `temperature_scale_mode`, and
`differentiate_temperature`. It also rejects the removed `reinforce_topk`
mode and discards `reinforce_*` fields. Use the new mode and `policy_*`
namespace above. Do not resurrect these removed names and break archived
experiments. Test that new fields survive YAML, CLI, and saved-config loading.

## 7. Repository integration plan

### 7.1 Gate implementation and forward-scoped records

Create a small dedicated implementation, for example
`src/wsparse/bottleneck/laplace_policy.py`, for sampling, row log density,
temperature scheduling, and typed policy records. Add a dispatch branch to
`AdaptiveLapSumTopKGate` in `src/wsparse/bottleneck/gate.py`; the class name
need not be refactored for this task.

Keep the old branches unchanged. The new branch must use ordinary
gather/mask/scatter autograd for values and differentiable gathered scores
for density. It must not call `rblapsum_gate`, `rblapsum_sf_gate`, barrier
solvers, `CarryChain`, or custom first-order backward machinery.

Collect each sampled row's differentiable log-density term and detached
diagnostics in a forward-scoped collector. It should be active only for an
opted-in training forward. Clear/destroy it in `finally` when the forward
ends. Do not retain live tensors in permanent diagnostic dictionaries or
append them across microbatches. Record instances, not only module names:
shared/repeated gates must not overwrite an earlier invocation.

Expose sampling as an explicit, temporary forward setting, independently of
recording policy terms. There are three necessary cases:

1. Training: sample supports and collect policy terms with grad enabled.
2. Stochastic evaluation: sample supports without collecting graphs.
3. Deterministic evaluation/probe/generation: do not sample or collect.

Do not toggle `model.train()` to request stochastic validation. Do not use
the existing `surrogate_active()` as the only switch: it returns false under
`no_grad`, which would make stochastic validation silently deterministic.
If temporary gate attributes or a context manager are used, restore them
on exceptions and nested calls. A forward collector must not become a
process-global persistent buffer.

### 7.2 Model loss interface

`TransformerLM.forward()` in `src/wsparse/model.py` currently returns
`(logits, loss)` and computes scalar mean CE. Preserve that default API and
the exact old-mode computation. Add an opt-in interface such as
`return_loss_details=True`, returning the extra policy records, unreduced
CE/sequence means, and counts needed by the new trainer branch. A typed
result is preferable to a positional collection of undocumented tensors.

Create and close the collector inside the model forward so DDP sees the
relevant graph among returned outputs. Do not compute a detached scalar CE
and attempt to reconstruct per-sequence costs afterward. Compute unreduced
float32 CE only in the new opt-in branch; preserve the old CE reduction
path for numerical regression checks.

The trainer should combine the returned CE and policy records once, then
call backward once. If the new training mode accidentally uses the legacy
scalar-only path without policy records, fail clearly rather than silently
train only the value gradient. Forward calls for inference or deliberate
value-only diagnostics may still use the standard two-result interface.

### 7.3 Wiring, scheduling, and code-residual support

`SparseTopKBottleneck` in `src/wsparse/bottleneck/module.py` explicitly passes
gate constructor parameters. Wire the new fields through this path; adding
config fields alone does not configure the gate.

`ActivationBottleneckController.set_step()` in
`src/wsparse/bottleneck/controller.py` is currently a no-op. Implement
policy-only scheduling there. `train()` currently does not call it in its
main step loop: add a call before the `on_step` probe and before any
microbatch, without perturbing old modes. Store runtime tau separately from
the configured initial value, so dumping config after annealing still saves
the original schedule. On resume compute tau from the restored step.

Support ordinary placements, multiple placements, shared projections,
tied decoders, and the current `code_residual` path. In
`TransformerLM._code_residual_stack()`, new gates should use the ordinary
`gate(code + update)` computation with no carry chain or split surrogate.
Later density losses reach the earlier carry through that sampled graph;
do not detach it to mimic an old first-order ablation.

Audit `effective_backward_support()` in `module.py`. Its current generic
non-hard fallback returns K+J for decoder-scale initialization. For the new
mode use K: the immediate decoder's value input/Jacobian has K sampled
active coordinates; the gate's own policy term branches through its scores,
not through a fictitious K+J-valued decoder input. Do not claim this is an
RMS calibration of the complete policy gradient. Preserve the old modes'
heuristic unchanged and test loader/initializer agreement.

### 7.4 Trainer, baseline state, and numerical checks

Add a new-mode branch in `src/wsparse/train.py`:

1. Set schedule for the current step; snapshot the previous-step baseline.
2. For each microbatch run one stochastic forward with details, build the
   computational loss, and accumulate/backpropagate it using the existing
   autocast/GradScaler/accumulation convention.
3. Keep real sampled CE for logging and detached CE sums/counts for the
   baseline update. Do not retain full logits or graph-bearing records
   after backward.
4. On the deterministic-probe cadence, run noise-off `eval()`/`no_grad()`
   forwards on the same step's cached integer input/target microbatches,
   before the optimizer update, using the unwrapped model. Cache only token
   tensors, not training graphs. Restore model mode and gate settings.
5. Check relevant values/gradients for finiteness, unscale and clip the full
   gradient as usual, then perform the optimizer update.
6. Update the EMA baseline once per step using detached statistics from all
   microbatches/ranks. Do not update it in gates, evaluation, or probes.

The deterministic training probe has dropout off, whereas the training CE
has ordinary training dropout. Label it as a probe; with nonzero dropout it
is not a controlled measurement of only support noise. Paired validation
has dropout off for both passes and supplies that controlled comparison.

Under DDP, call baseline collectives on every rank outside rank-0-only
logging/evaluation blocks. Rank-0-only diagnostic forwards should use the
unwrapped model, avoiding DDP forward collectives. Guard divergence through
the existing coordinated `should_stop` path; do not raise an exception on
one rank while others wait in gradient collectives.

Check finite stochastic CE, deterministic CE, effective widths, log-density
terms, computational loss, and gradients. `scripts/train_guard.py` currently
checks `train/ce` and `val/ce`; extend its nonfinite checks to the new CE keys
as appropriate. Preserve its existing CE thresholds and stop-step behavior.
Do not use support-term scalar values as a CE divergence threshold.

## 8. Evaluation, logging, and reproducibility

### 8.1 Required loss metrics

Preserve current key meanings and add explicit new ones:

| Metric | Meaning |
| --- | --- |
| `train/ce` | Actual sampled-support training CE; retain the existing guard key. |
| `train_stochastic/ce` | Alias of the same sampled CE, for explicit comparison. |
| `train_deterministic_probe/ce` | Periodic noise-off CE on the same pre-update training inputs. |
| `val/ce` | Clean deterministic Top-K CE, preserving cross-method comparisons. |
| `val_deterministic/ce` | Explicit alias of `val/ce`. |
| `val_stochastic/ce` | Monte Carlo mean CE with support sampling enabled. |
| `val_stochastic/mc_std` | Spread across repeated whole-evaluation draws; meaningful only with at least two draws. |
| `val_stochastic/samples` | Number of draws used. |

Add appropriate perplexities from actual CE. Preserve `train/loss` as a CE
alias if it remains in existing logs. Never exponentiate a policy
computational term or call it perplexity. If a support-term scalar is logged,
name it as a diagnostic and explain that its value is not an objective
quality metric. Do not report the zero-valued backward bracket as evidence
that selection learning is inactive.

`best_val_ce`, checkpoint metrics, summaries, and existing deterministic
final-validation keys must continue to refer to deterministic CE. Add
`val_final_stochastic/ce` and matching deterministic aliases at final
validation. Do not repurpose `val_soft/ce`: there is no soft forward here.

### 8.2 Paired validation protocol

Extend `evaluate()` or add a policy-aware wrapper without changing old
evaluation calls. Use the same deterministic token windows for both support
modes; the current evaluator already selects disjoint fixed windows by
`deterministic_offset`.

Keep dropout disabled for both modes. Evaluate stochastic draws at the
current scheduled temperature. For M draws, average actual CE over the same
validation batches in each draw, then average those M values; estimate MC
spread from the M whole-validation means, not from heterogeneous batch
losses. With M=1 omit or explicitly mark the MC spread unavailable; do not
present zero as measured uncertainty. One stochastic validation draw is one
additional forward per validation batch; M draws cost M such forwards.

Validation, deterministic probes, and generation must not change training
support RNG state, baseline state, schedule, or feature-usage statistics.
Use a separate evaluation generator and restore overrides with `try/finally`.
Fix evaluation seeds reproducibly across checkpoints to reduce comparison
noise. Never reseed each row identically, reuse the same noise vector for
all gates, or treat samples with different temperatures as identical laws.

### 8.3 Random generators and checkpoints

Use a dedicated training gate-noise generator on the active device, seeded
independently by rank from the run seed. Draw new independent noise for
every gate invocation, row, candidate, and accumulation microbatch. Global
Torch RNG alone would allow logging/validation to alter the training path.
Do not silently fall back to nondeterministic behavior if a backend cannot
support the requested generator; handle it explicitly and test CPU/CUDA.

Add a versioned optional checkpoint payload for policy training state,
including the EMA baseline and update count, and per-rank training noise
generator states. Restore it for new-mode resumes. Ensure all ranks
participate if gathering RNG states; a collective inside the existing
rank-0-only checkpoint branch would hang. Reject incompatible world-size
resumes unless an explicit documented reseeding policy is selected.

Keep this state out of required old-model state_dict keys so archived
checkpoints still load. `load_for_inference()` needs only saved config and
weights and should default to deterministic supports. Define a clear policy
for a new-mode training resume whose policy-state payload is missing;
prefer an error to silently resetting a baseline/RNG mid-anneal. Starting
from weights-only should be an explicit fresh-training operation.

Restoring gate RNG does not by itself guarantee bit-for-bit whole-run
resumption: training-data stream RNG and dropout/global RNG must also be
restored for that stronger claim. Audit the current resume implementation,
state the supported guarantee, and test gate continuation on fixed supplied
inputs at minimum. Do not advertise exact full-run continuation unless
those other states are also handled.

### 8.4 Per-gate diagnostics

At the existing diagnostics cadence log detached per-gate statistics, not
only an average across blocks:

- Scheduled tau; effective T mean/min/max; fraction at the width floor.
- Gamma mean/min/max, baseline, and detached advantage mean/RMS.
- Selected-index count, output-nonzero count, and candidate count.
- Fraction of sampled indices outside the clean Top-K but inside C;
  support overlap with clean Top-K; clean rank-boundary gap and span.
- RMS of the raw and scaled policy location gradient, including the
  projection when centering is enabled; score-gradient zero-sum residual.
- Finite-value failure indicators and total pre-clip gradient norm.

For raw-gradient diagnostics use detached analytic row formulas or hooks;
do not introduce a second training backward just to log them. Distinguish
score-space policy signals from total parameter gradients. Log what is
actually sampled, not CDF probabilities borrowed from RBLapSum.

## 9. Numerical implementation caveats

1. Generate noise, centered scores, widths, and density computations in at
   least float32 outside low-precision autocast. Casting scores to float32
   must retain their gradient. Values/decoder computations can follow the
   existing autocast dtype. No special float64 production path is needed,
   but use float64 for estimator tests.
2. A stable inverse-CDF sampler or a correctly implemented Laplace sampler
   is acceptable. Avoid log(0) at uniform endpoints. Endpoint clamps impose
   a machine-precision tail approximation: document it and verify sampled
   mean/variance and signs. Use independent variates, not one draw per row.
3. Use the identical realized detached noisy scores in both Top-K and
   density. Reconstructing a different sample or evaluating density only
   on selected features invalidates the estimator. Include all K+J sampled
   components, including discarded ones.
4. At extremely small T, finite-precision addition may round r back to u.
   Centering and float32 reduce this risk but do not eliminate it. Diagnose
   collapsed perturbations/ties and binding width floors; do not claim an
   arbitrarily small width is numerically accurate. Reject nonfinite scores
   instead of repairing them with a silent `nan_to_num`.
5. Density derivatives are location derivatives holding the sample fixed.
   This is not a case where fixed-noise finite differences of a hard
   forward must equal one sample's estimated gradient. Test the expected
   objective/gradient analytically or by Monte Carlo instead.
6. With detached T, `-log(2T)` contributes no parameter gradient, but include
   it in the actual density implementation. The zero-valued support bracket
   avoids an arbitrary large forward computational-loss value. Inf minus
   inf is still NaN: do not rely on subtraction to hide overflow.
7. Do not add entropy regularization, advantage clipping, adaptive same-batch
   normalization, antithetic multi-forward sampling, or an old RBLapSum
   correction without a separately specified objective/estimator. None is
   required to complete this version.
8. Existing selection-corrected initialization is calibrated for clean
   Top-K value statistics. Random support exchange can lower transmitted
   value energy even though cardinality stays K, particularly at large T.
   Keep the current initialization policy explicit; monitor per-block
   code/output RMS and the deterministic/stochastic CE gap. Do not silently
   add temperature-dependent decoder compensation, which would change the
   forward objective. A poor high-temperature forward can destabilize
   training independently of policy-gradient amplification.
9. Sequence-local credit assumes no cross-sequence forward coupling. That
   holds for this model's tokenwise RMS norms and within-sequence attention.
   If a later implementation adds batch-dependent normalization or mixes
   sequences, reassess which costs descend from each stochastic node;
   sequence CE alone would no longer generally be sufficient.

## 10. Relation to RBLapSum: explanation, not an extra implementation rule

For one gate with other noisy scores fixed, let t_i be the K-th largest
score among the other candidates. Integrating out epsilon_i gives the
conditional inclusion probability

$$
p_i=F\left(\frac{s_i-t_i}{T}\right),\qquad
\frac{\partial p_i}{\partial s_i}
=\frac{1}{2T}\exp\left(-\frac{|s_i-t_i|}{T}\right).
$$

Here F is the standard Laplace CDF. In the centered implementation this
statement can be interpreted using the equivalent uncentered noisy-score
rankings. The conditional cost mixture has derivative proportional to
C_i^+ - C_i^-, the actual CE change when i enters and the threshold candidate
leaves. A first-order value approximation to that swap uses g_i z_i - g_j z_j.
This connects the new objective to boundary-local Laplace kernels, but the
current deterministic-boundary RBLapSum VJP is not automatically identical
to the estimator or to its exact expectation.

The implementation must therefore sample supports and use CE-based density
terms. Do not replace the sampled score gradient by a CDF multiplier or a
`g*z*kappa` term while keeping the new method's name.

## 11. Tests and acceptance criteria

Add focused tests such as `tests/test_laplace_policy.py`, plus integration
coverage in existing config/trainer/DDP tests. Keep tests deterministic,
small, and CPU-capable. Use statistically justified tolerances for Monte
Carlo tests rather than brittle assertions on a handful of samples.

### 11.1 Forward and local autodiff

- Selected-index count is exactly K; all sampled indices belong to the
  current clean Top-(K+J) pool, including at deliberately large T.
- Signed transmitted values equal the corresponding original z entries.
  Noise and probabilities never rescale them.
- Clean noise-off output matches ordinary hard Top-K, including positive,
  negative, zero, and tied values under a specified tie convention.
- Stochastic `eval()` under `no_grad()` really samples; ordinary evaluation
  and generation do not. Overrides and exception restoration work.
- At fixed detached r, density autodiff matches sign(r-u)/T, and centered
  score gradients equal its projected version with zero sum.
- If r is intentionally left attached, demonstrate the incorrect zero
  location gradient; test the production path against this regression.
- Test that advantage, baseline, gamma, and relative T are detached, while
  the mean used for score centering is not.
- J=0 and K=N bypass density/noise; outputs and gradients are ordinary hard
  values. Reject J=1 with nontrivial relative-span mode.

### 11.2 Expected-gradient checks

Use a two-candidate K=1 toy with scores separate from fixed transmitted
values and support costs C_1, C_2. For independent Laplace(0,T) noises and
delta = s_1-s_2, the analytic policy derivative is

$$
\frac{\partial\mathcal J_T}{\partial s_1}
=(C_1-C_2)\frac{1+|\delta|/T}{4T}
\exp\left(-\frac{|\delta|}{T}\right),\qquad
\frac{\partial\mathcal J_T}{\partial s_2}
=-\frac{\partial\mathcal J_T}{\partial s_1}.
$$

Check Monte Carlo means at several score gaps and temperatures, for both
centering settings and valid baselines, with gamma=1. Test a parameter that
affects transmitted values as well: the combined ordinary-value and policy
gradient must match the derivative of the analytic expected loss.

For relative modes test the deliberately frozen-width partial derivative,
not a finite difference that recomputes the activation-dependent width.
For rescaling, check that the support path changes by the expected factor
while the hard value path is unchanged. In effective-temperature scaling,
the explicit T factor cancels the raw 1/T factor away from caps.

### 11.3 Multi-gate and trainer integration

- A small two-gate model agrees with an analytically enumerated support
  objective or a reliable expected-loss reference. The later density term
  has gradients to earlier continuous parameters. Detaching that path must
  fail the reference test.
- Cover stream and code-residual placements, shared projections, tied
  decoders, and multiple placements without double-counting or stale records.
- Check sequence cost weighting with ignored targets and variable valid
  lengths; accumulation scales both paths. Padding/ignored targets must not
  redefine credit as same-position CE only.
- EMA uses the previous step, stays constant across accumulation, updates
  once, and does not change during validation/probes.
- Validation records both actual CEs, uses identical data windows and
  dropout-off mode, and leaves training support RNG untouched. Verify final
  metrics, aliases, MC statistics, and deterministic best-val semantics.
- All schedule endpoints, hold/anneal transitions, relative row geometry,
  floors, rescaling modes, and resumed step values are tested.
- YAML, CLI overrides, `cfg.to_dict()`/`config_from_dict()`, saved JSON/YAML,
  and inference reconstruction preserve all new fields. Old archived
  configs/checkpoints still reconstruct unchanged.
- Dedicated RNG/baseline checkpoint continuation on fixed inputs reproduces
  subsequent supports and computational gradients. No graph retention
  grows across repeated forwards/backwards.
- Add an opt-in real two-process DDP test, including baseline collectives,
  per-rank RNG checkpointing, accumulation, and coordinated stopping.

### 11.4 Repository verification

Run the normal suite with the known flaky test deselected and report that
exclusion, as required by `AGENTS.md`. Run opt-in DDP tests on a suitable
environment. Because this work touches trainer/model/config paths, run
`tools/repro_check.py` against the previous commit for existing regimes,
including hard, LapSum, and RBLapSum references, using the tool's actual CLI.
Do not alter old-mode CE or gradients as an incidental refactor.

Perform a real short smoke of a shipped old config and the new configs on
a user-provided GPU box. Stop through `train_guard.py --stop-step`; never
shorten `train.max_steps`, which would change LR/temperature trajectories.
Use absolute, relative, rescaling-on/off, and code-residual cases. Verify
both CE metrics, finite policy gradients, actual support exchanges, and
resume state. A short smoke is a plumbing check, not evidence of improved
convergence. Record limitations and negative results plainly.

## 12. Shipped configuration examples

Add complete inheriting configs, for example
`configs/bn_laplace_policy_absolute.yaml` and
`configs/bn_laplace_policy_relative.yaml`, based on a currently valid shipped
architecture. Do not modify existing campaign files. The following fragments
illustrate new fields; adjust schedule length to the inherited max_steps and
preserve appropriate existing projection initialization.

Reference absolute-temperature example:

```yaml
activation_bottleneck:
  surrogate_mode: laplace_policy
  selection_mode: abs_topk
  temperature: 0.5
  policy_temperature_mode: absolute
  policy_temperature_schedule: exponential
  policy_temperature_final: 0.05
  policy_temperature_hold_steps: 1000
  policy_temperature_anneal_steps: 17000
  policy_min_temperature: 1.0e-6
  policy_center_scores: true
  policy_baseline: ema
  policy_baseline_decay: 0.99
  policy_support_scale: 1.0
  policy_support_scale_mode: constant

train:
  compile: false
  policy_val_samples: 2
  policy_val_seed: 1337
  policy_train_deterministic_every_steps: 0
```

Relative-span example with an explicitly modified support gradient:

```yaml
activation_bottleneck:
  surrogate_mode: laplace_policy
  selection_mode: abs_topk
  temperature: 0.5
  policy_temperature_mode: relative_span
  policy_temperature_schedule: exponential
  policy_temperature_final: 0.05
  policy_temperature_hold_steps: 1000
  policy_temperature_anneal_steps: 17000
  policy_min_temperature: 1.0e-6
  policy_center_scores: true
  policy_baseline: ema
  policy_baseline_decay: 0.99
  policy_support_scale: 0.1
  policy_support_scale_mode: scheduled_temperature
  policy_support_temperature_ref: 0.5
  policy_support_scale_max: 0.1

train:
  compile: false
  policy_val_samples: 2
  policy_val_seed: 1337
  policy_train_deterministic_every_steps: 0
```

The second configuration freezes the row scale and multiplies the policy
gradient by a decreasing gamma. It is not the reference unbiased full
expected-CE gradient. To cancel the complete sampled 1/T factor instead,
select `effective_temperature` and specify a reference width in score units.
The numerical coefficients above are illustrative, not recommended optimal
values for every K, J, placement, or encoder scale.

## 13. Final report from the implementing agent

Report the implemented objective and deliberate approximations, files
changed, exact config names/defaults, training/evaluation costs, test and
repro results, and any unverified GPU/DDP behavior. Include stochastic and
deterministic CE from smoke runs without asserting a convergence improvement.
Confirm that baseline, RNG, schedule, and old-mode compatibility were tested.
Do not mark the task complete if only a gate helper exists without trainer,
evaluation, checkpoint, and configuration integration.

Follow the repository's commit/push rules only after its required checks
pass. Do not start a training campaign, rent a box, change existing runs, or
introduce extra objectives merely to complete this implementation handoff.
