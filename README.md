# weight-sparsity

Training small language models on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), with
a **sparse activation bottleneck** and differentiable surrogate gradients for
the hard TopK selection it makes:

```
x ──▶ W_in ──▶ TopK / AbsTopK  (exactly K of N) ──▶ W_out ──▶ …
```

`W_in` and `W_out` are ordinary dense `nn.Linear` layers trained by the model's
ordinary objective: no reconstruction loss, no weight mask, no pruning. What is
sparse is the **code** between them, and the research question is what gradient
to give a selection that the forward pass makes discontinuously.

| `surrogate_mode` | forward | backward |
| --- | --- | --- |
| `hard` | exact hard TopK | hard mask only; the `J` extra candidates get nothing |
| `lapsum` | exact hard TopK | LapSum Top-(K+J) VJP at constant `T`, budget barrier `Σpᵢ = K` |
| `rblapsum` | exact hard TopK | local kernel at the **hard rank boundary** `b = max(b₀, s_(K+1))` |
| `rblapsum_sf` | `yᵢ = zᵢ·pᵢ` ("soft forward") | the gradient of that forward |
| `laplace_policy` | exact hard TopK on a **sampled** support: Laplace noise on the Top-(K+J) scores, the `K` largest noisy scores kept, original values transmitted | hard-mask value gradient plus the likelihood-ratio (score-function) gradient of the sampled CE ([note](docs/laplace-policy-topk.md)) |

The repo name is historical. The weight-sparsity subsystem (LTP, Continuous
Sparsification, TopK weight masks) was **removed on 2026-09-28**, along with the
adaptive/scheduled temperature machinery, the reinforce/jumprelu/swap-Gibbs
surrogates, the reconstruction loss and the output calibration. Archived
`config.json` files still load: `wsparse.config` migrates the keys it can (see
[Loading archived runs](#loading-archived-runs)) and raises a clear error on the
ones whose behaviour no longer exists.

## Install

```bash
git clone https://github.com/labofdoubt/weight-sparsity.git
cd weight-sparsity
pip install -e .[data]          # torch, numpy, pyyaml, datasets, transformers, tokenizers
```

## Quick start

```bash
# 1. download + tokenize TinyStories into data/tinystories/{train,val}.bin
python -m wsparse.data --config configs/bn_dense.yaml

# 2. train one of the three matched setups
python -m wsparse.train --config configs/bn_dense.yaml    # no bottleneck
python -m wsparse.train --config configs/bn_hard.yaml     # + hard TopK bottleneck
python -m wsparse.train --config configs/bn_lapsum.yaml   # + LapSum surrogate gradient

# 3. inspect / sample
python scripts/model_summary.py --config configs/bn_hard.yaml
python scripts/generate.py --ckpt runs/bn_hard/latest.pt
```

Any field can be overridden from the command line:

```bash
python -m wsparse.train --config configs/bn_lapsum.yaml \
    --model.n_layers=16 --train.lr=3e-4 \
    --activation_bottleneck.surrogate_mode=rblapsum \
    --activation_bottleneck.k=32 --activation_bottleneck.j=480 \
    --activation_bottleneck.temperature=2.0
```

Configs compose through a `_base_` key (see `configs/*.yaml`); fragments live in
`configs/model`, `configs/train` and `configs/mdinit`.

Multi-GPU runs go through `torchrun`; `scripts/train_ddp.sh` wraps it (one
guarded process per GPU), and
`configs/fineweb_rbk_500m.yaml` is the 8-GPU FineWeb-Edu configuration
(see [Multi-GPU](#multi-gpu-ddp)).

## What gets printed during training

`configs/bn_lapsum.yaml` with `--activation_bottleneck.surrogate_mode=rblapsum`:

```
[train] device=cuda:0 dtype=torch.bfloat16 params=101.3M (non-emb 68.8M) batch=64x512 tok per rank (micro 16 x accum 4)
[train] param groups: 2
[train] activation bottleneck: 10 layers (all) residual_out N=1536 K=32 J=480 (abs_topk, rblapsum) density=0.021 params=19.7M
step    200/20000 | loss 3.9214 | ce 3.9214 | ppl   50.47 | lr 6.00e-04 | L0 32.0 | cap 0.98 \
 | gnorm 0.51 | 41.2K tok/s | 124 ms/step
step    500 | val ce 3.7410 | val ppl 42.15
```

The bottleneck fields depend on the mode: `t` and `dK` (`|Σp − K|`) for the
barrier modes, `L0` (the realized hard support size) and `cap` (how often the
rank cap rather than the floor `b₀` sets the boundary) for the rblapsum modes,
`gap` (`r_K − r_{K+1}`) for the hard baseline, which solves nothing.

Everything is also written to `runs/<run_name>/metrics.jsonl`, to TensorBoard
under `runs/<run_name>/tb` (`train.tensorboard`, on by default), and to Weights
& Biases if `train.wandb_project` is set. Metric names are already namespaced
with `/`, which is TensorBoard's grouping convention, so pointing it at the
parent directory overlays every run in one chart:

```bash
pip install tensorboard          # or: pip install -e .[logging]
tensorboard --logdir runs
```

Generated samples are text, not scalars, so they go to `runs/<run_name>/samples.txt`
and to TensorBoard's TEXT tab (and wandb) rather than into `metrics.jsonl`.
`sample_count` continuations are drawn per event as one batch, from a generator
seeded on `train.seed + step` — without that, sampling would draw from the
global RNG, whose state depends on everything the run consumed beforehand, so
samples would not be comparable across runs once any dropout is enabled.

| metric | meaning |
| --- | --- |
| `train/ce`, `train/loss` | cross-entropy (identical: there is no auxiliary loss term) |
| `train/lr`, `train/grad_norm` | learning rate, pre-clip gradient norm |
| `perf/tokens_per_s`, `perf/ms_per_step`, `perf/tokens_seen` | throughput; `tokens_per_s` is global across ranks |
| `val/ce`, `val/ppl` | validation loss, at `hard_inference` (so comparable across modes) |
| `val_soft/ce` | `rblapsum_sf` only: the same batches through the soft forward |
| `train_stochastic/ce`, `train_deterministic_probe/ce` | `laplace_policy` only: `train/ce` is the sampled-support CE (the alias says so); the probe is a periodic noise-off forward on the same training inputs, before the update |
| `val_deterministic/ce`, `val_stochastic/ce`, `val_stochastic/mc_std` | `laplace_policy` only: `val/ce` stays the clean Top-K forward (alias `val_deterministic`); the stochastic pass samples the supports on the same windows, `train.policy_val_samples` draws, spread only from two draws on |
| `policy/tau`, `policy/baseline`, `policy/support_term_diag` | `laplace_policy` only: the step's scheduled width parameter, the EMA baseline used, and the detached support term (a gradient diagnostic, not a loss) |
| `val_final/ce` | the larger end-of-run evaluation (`train.final_val_batches`) |
| `bottleneck/density`, `candidate_density` | `K/N` and `(K+J)/N` |
| `bottleneck/temperature` | the constant `T` (flat by construction; logged so a run's own record carries it) |
| `bottleneck/barrier`, `barrier_gap`, `frac_above_barrier` | `b`, `r_{K+1} − b`, and how often the barrier sits below `r_{K+1}` |
| `bottleneck/budget_residual`, `barrier_failures` | `\|Σp − K\|`, and how often it exceeds the achievable-precision floor |
| `bottleneck/score_gap`, `score_span` | `r_K − r_{K+1}` and `r_K − r_{K+J}` |
| `bottleneck/active_count`, `active_count_max` | rblapsum: the realized hard support size (`≤ K`) |
| `bottleneck/rb_boundary`, `rb_b_rank`, `rb_cap_active_frac` | rblapsum: the boundary, its rank, and how often the rank cap wins over `b₀` |
| `bottleneck/rb_soft_mass` | `rblapsum_sf` only: `Σᵢ pᵢ`, the forward's soft L0 |
| `bottleneck/rb_support_grad_norm`, `rb_common_mode`, `rb_kick_win`, `rb_boundary_grad_ratio` | rblapsum: size and structure of the support gradient `g_s` |
| `bottleneck/policy_t_mean`, `policy_t_floor_frac`, `policy_gamma_mean`, `policy_exchange_frac`, `policy_overlap`, `policy_score_grad_rms`, `policy_collapsed_frac`, … | `laplace_policy`: the effective noise width and how often its floor binds, the support multiplier, the fraction of sampled members outside the clean Top-K, the RMS of the sampled location gradient `sign(r − u)/T`, perturbations that rounded away; also per gate as `bottleneck_policy_*/blocks.i` |
| `bottleneck/grad_active`, `grad_inactive`, `grad_rank_bin0..7` | surrogate gradient magnitude on the active `K`, the exploratory `J`, and by candidate rank |
| `bottleneck/feature_dead_frac`, `feature_usage_entropy`, `feature_usage_max` | **feature collapse — see [Diagnostics](#diagnostics)** |
| `bottleneck_<key>/<layer>` | the same keys per layer, for the TensorBoard panels |

## Configuration

### `model`

| field | default | notes |
| --- | --- | --- |
| `n_layers`, `d_model`, `n_heads` | 12 / 768 / 12 | |
| `mlp_ratio` | 4.0 | `d_mlp = round(mlp_ratio · d_model)`, rounded up to a multiple of 8 |
| `mlp_activation` | `gelu` | `gelu`, `relu`, `silu`, `swiglu` |
| `max_seq_len` | 512 | learnable positional embedding table size (`pos_encoding: learned`) |
| `pos_encoding` | `learned` | `learned` or `rope` (`rope_theta`), in which case there is no position table |
| `bias` | `false` | no biases anywhere by default |
| `norm_eps` | 1e-6 | RMSNorm (pre-norm blocks, plus a final norm) |
| `tie_embeddings` | `true` | shares `lm_head` with the token embedding |
| `decouple`, `decouple_gains` | `false`, `row_col` | magnitude-direction decoupling, below |
| `md_init` | `false` | MD's initialization without its optimizer (the ablation) |
| `init_scheme` | `fixed_std` | `fixed_std`: `w ~ N(0, init_std²)`; `fan_in`: `w ~ N(0, init_gain²/fan_in)` |
| `init_std`, `init_gain` | 0.02, 1.0 | weight standard deviation / fan-in gain |
| `init_std_embedding`, `init_std_pos` | `null` | embedding stds; default to `1/√2` each, so `tok_emb + pos_emb` has unit variance per element at init. `init_std_pos` falls back to `init_std_embedding` when that is set. Never affected by `init_scheme` |
| `init_std_unembedding` | `null` | the unembedding (`lm_head`) std, untied only — tied, there is one matrix and `init_std_embedding` sets it (setting both raises). Defaults to `1/√d_model`, putting the init logits at unit std. Independent of `init_scheme` / `init_std` / `init_gain`, which never reach the head |
| `init_scale_residual` | `true` | scales every residual output projection by `1/√(2·n_layers)` |
| `logit_scale` | `auto` | `auto`: divide the logits by `unemb_std · √d_model`, normalizing them to ~unit std. A no-op at the untied default (already 1); it matters for a tied head, whose `1/√2` std would otherwise put init logits at std ≈ 20. `none`: no rescaling |
| `dropout`, `attn_dropout` | 0.0 | |

Sizes: `configs/model/{tiny,small,mid80,medium,large}.yaml` →
3.1M / 25.2M / 49.2M / 85.0M / 113.3M non-embedding parameters (16.1M / 51.2M /
81.7M / 123.9M / 152.3M with the 50257-token GPT-Neo vocabulary, tied).

### `data`

`tokenizer: gpt_neo` (default) uses the GPT-Neo/GPT-2 byte-level BPE, i.e. the
tokenizer the original TinyStories models were trained with. `tokenizer: bpe`
instead trains a small byte-level BPE on TinyStories itself
(`bpe_vocab_size`, default 8192), which shrinks the embedding matrix a lot and
puts more of the parameter budget into the transformer body.

`scripts/prepare_fineweb.py` writes the same `{train,val}.bin` + `meta.json`
layout from FineWeb-Edu `sample-10BT`, holding out a document-aligned 50M-token
validation split (9.90B train / 50.0M val tokens at `gpt_neo`).

### `train`

`optimizer` (AdamW), `lr`, `betas`, `eps`, `weight_decay`, `grad_clip`,
`lr_schedule` (`cosine`/`linear`/`constant`), `warmup_steps`, `min_lr_ratio`,
`batch_size` (sequences per optimizer step, **per rank**), `micro_batch_size`
(gradient accumulation = `batch_size / micro_batch_size`), `max_steps`,
`log_every_steps`, `validate_every_steps`, `val_batches`, `final_val_batches`,
`checkpoint_every_steps`, `keep_last_checkpoints`, `sample_every_steps`,
`sample_prompt`, `sample_tokens`, `sample_count`, `seed`, `device`, `dtype`,
`compile`, `out_dir`, `run_name`, `resume`, `tensorboard`, `wandb_project`,
`wandb_entity`.

Weight decay is applied only to ≥2D weights; RMSNorm gains and biases are
excluded. Under `model.decouple` the optimizer is the decoupled one instead and
`train.weight_decay` is ignored (the banner says so).

`train.compile` stays `false`: the gate's custom autograd functions graph-break,
and the compiled path has never been validated against eager for them.

### `activation_bottleneck`

```yaml
activation_bottleneck:
  enabled: true
  layers: all              # all | even | odd | first:n | last:n | [0, 2, 4]
  placement: residual_out  # pre_mlp | residual | residual_out | post_attn | post_mlp
  n_features: 1536         # N
  k: 32                    # K, active in the forward pass
  j: 480                   # J, extra candidates that only receive gradient
  selection_mode: abs_topk # topk | abs_topk | gated_topk
  surrogate_mode: rblapsum # hard | lapsum | rblapsum | rblapsum_sf | soft_ste | laplace_policy
  temperature: 1.0         # the one kernel/barrier bandwidth, in score units (laplace_policy: tau_0)
  # --- rblapsum family ---
  rblapsum_boundary_grad_mode: null   # null -> detach (rblapsum) / through_rank_kappa (sf)
  rblapsum_boundary_floor: null       # b0; null -> 0.0, no floor
  rblapsum_support_scale: 1.0         # scales only the support-exchange term g_s
  rblapsum_rho_random_perm_prob_grad: 0.0   # signal-permutation ablation
  rblapsum_sf_value_grad: pool        # pool | support
  # --- laplace_policy (see docs/laplace-policy-topk.md; configs/bn_laplace_policy_*.yaml) ---
  policy_temperature_mode: absolute   # absolute | relative_b | relative_span (relative scales detached)
  policy_temperature_schedule: constant   # constant | exponential (+ _final, _hold_steps, _anneal_steps)
  policy_min_temperature: 1.0e-6      # numerical width floor, in score units
  policy_center_scores: true          # centre the pool's scores before the noise (differentiable mean)
  policy_baseline: ema                # none | ema (previous steps' mean CE; _decay, _initial)
  policy_support_scale: 1.0           # gamma_0 on the selection gradient only; _mode constant |
                                      # effective_temperature | scheduled_temperature, _temperature_ref, _max
  # --- shape and init ---
  post_norm: false         # an RMSNorm on each bottleneck's own output
  init_mode: default       # default | sqrt_k | sqrt_k_selection_corrected | unit_norm_dictionary
  tie_encoder_decoder: false
  share_projections: false # one encoder and one decoder for every installed bottleneck
  bias: false              # biasless projections (the family convention)
  # --- numerics ---
  barrier_solver_tol: 1.0e-6
  solver_dtype: float32
  log_diagnostics: true
  hard_inference: true     # skip the soft-mask machinery outside training
```

Static validation covers the obviously impossible: `1 ≤ K < N`, `K+J ≤ N`, and
`J ≥ 1` for every mode that has a surrogate (`hard` and `laplace_policy` accept
`J = 0`; `hard` ignores `J` beyond the diagnostics, which is why the archived
hard runs carry a nominal `j`, and `laplace_policy` with no possible exchange
is the hard gate).  `train.policy_val_samples`, `policy_val_seed` and
`policy_train_deterministic_every_steps` configure the `laplace_policy`
evaluation.

## Activation bottleneck

`placement` decides what the bottleneck sits on. `pre_mlp` bottlenecks
`self.norm2(x)`, the pre-RMSNorm MLP input, so the residual stream itself stays
dense and the skip routes around the bottleneck. `post_attn` / `post_mlp`
constrain what a branch may *contribute* instead of what it may read.
`residual` / `residual_out` replace the stream itself at the head or the tail of
the block, so nothing routes around them — which is the regime where the
depth-compounding effects in `docs/` show up.

`share_projections: true` installs the bottlenecks as usual but gives them one
encoder and one decoder: `in_proj` / `out_proj` (and `score_proj` under
`gated_topk`) are the same objects in every module, so the parameter cost is one
bottleneck's worth however many layers and placements are selected, and each
matrix receives the sum of every bottleneck's gradient. Gates and their usage
statistics, `post_norm` gains and the decoder scale stay per bottleneck. The
state dict lists the shared matrices under every bottleneck's prefix, all
copies identical, so checkpoints load either way. `tie_encoder_decoder`
composes with it and leaves a single matrix for the whole stack.

### Forward: exact hard TopK

With ranking score `r = a` (`topk`) or `r = |a|` (`abs_topk`):

```
â_i = a_i · 1[i ∈ TopK_K(r)]
```

Selection is by magnitude in `abs_topk`, but the **signed** activation is
forwarded — never `|a_i|`. There is no ReLU before the TopK.

`selection_mode: gated_topk` splits the two jobs the single projection
otherwise does, with independent branches:

```
s = W_s x + b_s      ranks the support        y_i = 1[i ∈ TopK_K(s)] · v_i
v = W_v x + b_v      carries the value
```

The support depends on `s` alone — never on `v`, `|v|` or `s·v` — and no
nonlinearity is applied to `v`, so values stay signed. The gradients then
separate exactly: `∂L/∂v = m ⊙ g` (only selected features get value updates,
the exact hard-mask gradient) while `∂L/∂s` is the constrained LapSum VJP on
`uᵢ = gᵢvᵢ`, so an inactive feature can still learn to raise its score and enter
the support. That falls out of the same `m_hard + (p − stopgrad(p))` mask
applied to a *separate* value tensor, so it reuses the existing VJP rather than
re-deriving the Jacobian. It costs one extra `d_model × n_features` projection
per layer.

Except under `rblapsum_sf`, the forward pass is exactly `K`-sparse no matter
what `J` or the temperature are set to; the soft probabilities below are never
used numerically in the forward pass.  (`laplace_policy` is exactly `K`-sparse
too, on a support sampled from the Top-(K+J) pool during training; see the
[implementation note](docs/laplace-policy-topk.md).)

### Backward: LapSum over Top-(K+J)

One `torch.topk(K+J, sorted=True)` per call produces the sorted candidate pool
that the hard mask, the barrier solve, the probabilities, the backward and the
diagnostics all share. Its first `K` entries are the active set `A`; the
remaining `J` are inactive candidates. Everything outside Top-(K+J) receives
exactly zero gradient from this module.

Over that pool, with the Laplace CDF `F(z) = ½eᶻ` for `z ≤ 0` and `1 − ½e⁻ᶻ`
otherwise, the soft mask is `pᵢ = F((rᵢ − b)/T)` with the barrier `b` fixed by
`Σᵢ pᵢ = K`. The mask handed to the layer is

```
m = m_hard + (p − stopgrad(p))
```

so `m == m_hard` numerically while `∂m/∂r == ∂p/∂r`. Writing `κᵢ = φᵢ/T`,
`uᵢ = gᵢaᵢ` and `q^budget = κ/Σκ`, the exact fixed-`T` VJP is

```
∂L_mask/∂rᵢ = κᵢ · (uᵢ − ⟨q^budget, u⟩)
```

The subtraction is not optional: `b` is a function of *every* candidate score
through the budget constraint, and `∂b/∂r_l` is exactly `q^budget_l`. Dropping
it — using `κᵢuᵢ` — would let the surrogate inflate the budget instead of
trading candidates against one another. `tests/test_bottleneck.py` checks this
against finite differences **with the barrier re-solved at each perturbation**,
which is what makes the correction observable; the hard TopK discontinuity is
never finite-differenced.

### The temperature is a constant, detached bandwidth

`T` is `activation_bottleneck.temperature`: one positive number, in raw score
units, shared by the `lapsum` and `rblapsum` kernels. It is not learned, not
scheduled, not score-relative and not solved for.

> The temperature is a **detached bandwidth**. The backward is the exact
> surrogate VJP conditional on that fixed bandwidth.

Nothing differentiates through a solver, and the only score dependence carried
into the backward is the boundary's: `b = b(r; T)` via `Σpᵢ = K` for `lapsum`
(the `⟨q^budget, u⟩` term), or the rank boundary's own derivative for
`rblapsum` (`rblapsum_boundary_grad_mode`, below).

This unification replaced three earlier mechanisms — an `n_eff`-calibrated
adaptive temperature, a prescribed schedule, and a score-relative scale mode —
all removed on 2026-09-28. `docs/relative-temperature-divergence.md` records
why the relative mode was abandoned; the constant-`T` runs it was compared
against are the ones the current code reproduces bit for bit.

### The closed-form barrier

`Σᵢ F((rᵢ−b)/T) = K` is solved in closed form, not by iteration. On the interval
`r_j ≥ b ≥ r_{j+1}` the budget is `j − ½e^{b/T}A_j + ½e^{−b/T}B_j`, so with
`y = e^{b/T}`:

```
A_j y² + 2(K−j) y − B_j = 0,     A_j = Σ_{i≤j} e^{−rᵢ/T},  B_j = Σ_{i>j} e^{rᵢ/T}
```

Because the candidates are already sorted, one `logcumsumexp` prefix scan and one
suffix scan give every `A_j`/`B_j` in log space; the budget at each knot is
increasing in `j`, so the interval index is just `Σ_j 1[budget(r_j) ≤ K]`; and the
positive root is taken as

```
log y = ½(log B − log A) + asinh( (j−K)·e^{−½(log A + log B)} )
```

which is the branch-free form of the quadratic root — it avoids the cancellation
that `−B + √(B²+4AC)` suffers for `j < K`, and its large-argument limit
reproduces the `A = 0` and `B = 0` edge intervals exactly. No second sort, no
bisection. `lapsum_barrier_bisect` is the slow reference used to validate it.

Two numerical details that are load-bearing, both found by stress testing:

* The scans run in coordinates centred on `r_K` and clamped to `±60` (float32).
  `F` saturates past `|z| ≈ 40`, so the clamp is numerically free, but without
  it a single score sitting 10⁸ temperatures away leaves `log A` with no
  significant digits at all (`1e8 − 1e8` in float32) — that produced a budget
  residual of **93** against `K = 32` before it was fixed. Anchoring at `r_K`
  rather than `r_max` is what makes the clamp safe: whenever the span/`T` ratio
  is large enough for it to bite, `Σp = K` forces `b` into the `r_K`/`r_{K+1}`
  gap.
* The `barrier_failures` diagnostic compares against an achievable-precision
  floor, not a flat tolerance. Scores arrive already rounded, so `(rᵢ − b)`
  carries `~eps·|r|` of error that `1/T` amplifies; a flat `1e-6·K` threshold
  reports a failure on every batch of offset activations while the solver is in
  fact exact.

### RBLapSum: the boundary is the hard rank

`surrogate_mode: rblapsum` keeps LapSum's hard forward, its Top-(K+J) pool and
its Laplace kernel, but replaces the soft-mass constraint with the **hard rank**
boundary

```
b = max(b₀, s_(K+1)),        κᵢ = e^{−|sᵢ − b|/T} / (2T)
```

so `K` becomes an upper cap on the active count rather than an exact budget
(`b₀ = rblapsum_boundary_floor` is a fixed activation floor; where it binds,
fewer than `K` features are active and `rb_cap_active_frac` falls below 1).
There is no target-count penalty and no budget solve.

`rblapsum_boundary_grad_mode` decides what the boundary's own derivative
contributes:

| mode | boundary term |
| --- | --- |
| `detach` (default) | none — independent local gradients per candidate |
| `project` | `detach`, minus the common-mode score direction (cap-active rows only) |
| `through_rank` | differentiate through `s_(K+1)`, which *is* the boundary |
| `through_rank_kappa` | the same zero-sum correction, distributed `κ`-weighted |

`through_rank` at a sharp `T` concentrates the compensation on the boundary
feature alone and the boundary score can run away (measured: `j=32, T=1`
diverged around 1.6k steps on both seeds). `through_rank_kappa` spreads it in
LapSum's rank-one Jacobian form and removes the runaway —
`docs/rblapsum-stabilization.tex`, and `docs/rblapsum-kappa-ablation.tex` for
the ablation. `rblapsum_support_scale` multiplies that support-exchange term
`g_s` after the mode correction, so `0.0` is exactly the hard-TopK backward and
`1.0` the unmodified surrogate.

`rblapsum_rho_random_perm_prob_grad` is the signal-permutation ablation: a `ρ`
fraction of each row's `dL/dpᵢ` values is shuffled *before* the kernel
weighting, which preserves the surrogate's scale profile, locality and zero-sum
structure while destroying the assignment of signal to neuron.
`ρ = 0` is bitwise the unmodified backward (`docs/rblapsum-signal-permutation.tex`).

### RBLapSum soft forward

`surrogate_mode: rblapsum_sf` puts the probabilities **in** the forward:
`yᵢ = zᵢ·pᵢ` over the Top-(K+J) pool, exactly `0` outside it, and the backward is
the gradient of that forward — no train-time forward/backward discrepancy. At
`through_rank` with `rblapsum_support_scale = 1.0` it is plain autograd.

Evaluation still runs the hard Top-K forward while `hard_inference` is set, so
`val/ce` stays comparable across modes; the soft forward's own loss is logged
separately as `val_soft/ce`. `rblapsum_sf_value_grad: support` masks the value
path to the hard support, so the `J` inactive candidates train their ranking but
not their content (`docs/rblapsum-soft-forward.tex`,
`docs/rblapsum-sf-support-values.tex`).

### Magnitude-direction decoupling

`model.decouple` trains every matrix as a direction on a fixed-norm sphere plus
an explicit gain (`decouple_gains: row_col` gives each row and column its own),
with `md_init_` re-initializing the matrices onto their spheres first.
`model.md_init: true` applies that initialization alone, with the ordinary
AdamW, which is the ablation that separates the initialization from the
optimizer (`docs/md-init-vs-decoupling.tex`). `train.weight_decay` does not
apply under `decouple`.

### Diagnostics

Per bottleneck layer, averaged across layers in the logs — see the metric table
above for the full list. The failure counters are first-class: `barrier_failures`
counts rows whose budget residual exceeds what the solver dtype can achieve,
rather than a flat tolerance.

**`feature_dead_frac` and `feature_usage_entropy` deserve particular attention.**
The characteristic failure of a TopK activation bottleneck is collapse: a subset
of the `N` features wins every token, the rest are never selected, and their
`W_in`/`W_out` columns stop receiving gradient entirely — so the effective width
is far below `N`. The loss and the budget residual both look perfectly healthy
while that happens, so nothing else logged here would reveal it. Usage is
tracked as a bias-corrected EMA (a uniform-seeded one would take ~460 steps to
decay past the dead threshold, reporting 0% dead throughout the early phase when
collapse is most likely). `feature_usage_entropy` is `exp(H)/N`: 1.0 is even
usage, and it falls to the surviving fraction — on a synthetic collapse where 64
of 256 features take every slot, it reads 0.250 and `feature_dead_frac` reads
0.750, both from step 10.

`scripts/dead_feature_watchdog.py` watches that live, and `analysis/` holds the
offline probes (score ladders, gradient probes, the streamlit score explorer).

### Cost

Gate only — no projections, no model — 4096 rows at `N=2048, K=256, J=768,
T=1.0`, float32 CPU, forward + backward:

| variant | ms | vs hard TopK |
| --- | --- | --- |
| hard TopK backward | 46 | 1.00× |
| LapSum (budget barrier) | 245 | 5.29× |
| RBLapSum, `detach` | 121 | 2.62× |
| RBLapSum, `through_rank_kappa` | 126 | 2.72× |
| RBLapSum soft forward | 137 | 2.97× |

The absolute numbers are machine-specific (these are one laptop's CPU); the
ratios are the point. RBLapSum is the cheaper surrogate because it needs no
barrier solve — its boundary is an index into the sorted pool. End-to-end the
bottleneck itself dominates: it adds two `d_model × N` projections per
bottlenecked layer, and none of the surrogate cost is paid at inference, where
`hard_inference` skips the machinery entirely.

## Multi-GPU (DDP)

```bash
NGPU=8 bash scripts/train_ddp.sh --config configs/fineweb_rbk_500m.yaml \
    --train.run_name=my_run
# = torchrun --standalone --nproc_per_node 8 scripts/train_guard.py --config ...
```

`train()` detects `torchrun`'s environment and wraps the model in `DDP`; a
single-process launch sees `world == 1` and takes none of those branches.
What is per-rank is deliberate:

* `train.batch_size` is **per rank**, so the global batch is
  `world × batch_size` sequences and the throughput metrics report global
  tokens.
* The training stream's seed is offset by rank (`seed + 7919·rank`), so each
  rank draws different batches. Model init uses the same seed on every rank.
* `broadcast_buffers=False`: the gates' usage EMAs are diagnostics, and
  synchronizing them every step would cost a collective for nothing.
* Logging, validation, sampling, checkpointing and the dumped config are rank 0
  only.
* Stopping is coordinated by an all-reduced flag (`scripts/train_stop.py` /
  `train_guard.py --stop-step`), not by an exception — an exception on one rank
  would hang the others in their next collective.

The configured device decides the backend: `cuda` → NCCL with `cuda:local_rank`
per rank, `cpu` → gloo (which is what the 2-process test in
`tests/test_train_ddp.py` uses, under `WSPARSE_DDP_TEST=1`).

## Loading archived runs

`config_from_dict` / `load_config` migrate configs written before the cleanup:

* a `sparsity` block is dropped when disabled, and **raises** when
  `sparsity.enabled` was true;
* `lapsum_fixed` (absolute) → `lapsum` with `temperature = fixed_temperature`;
* a constant-schedule `lapsum_scheduled` → `lapsum` with
  `temperature = temperature_start`;
* `rblapsum_temperature` → `temperature`;
* every removed knob (`n_eff`, `effective_count_metric`, `boundary_mode`,
  `one_sided_weight_mode`, the `temperature_*` schedule fields,
  `surrogate_grad_scale`, `inactive_grad_scale`, `project_scale_gradient`, the
  servo fields, `reconstruction_*`, `calibrate_output`, …) is dropped;
* the modes whose behaviour is gone — `lapsum_adaptive`, a non-constant or
  score-relative `lapsum_scheduled`, `swap_gibbs`, `jumprelu`,
  `reinforce_topk`, a temperature servo, a non-zero `reconstruction_coef`,
  `calibrate_output` — raise with a message naming the removal.

`tools/repro_check.py` is what proved the surviving modes unchanged: it runs a
tiny real training job under deterministic kernels and digests per-step losses,
validation CEs, probe logits, every parameter hash and the optimizer state. Run
against the pre-cleanup code (old schema) and the cleaned code (new schema), it
reports **IDENTICAL** for `hard`, `lapsum` at constant `T`, and `rblapsum` with
`through_rank_kappa`.

## Notebooks

Three matched runs (`SETUP = 'dense' | 'hard' | 'lapsum'`), with a parameter
table, shared hyper-parameters and seed, loss / surrogate-gradient / feature-usage
curves and a comparison table:

* `notebooks/colab_bottleneck.ipynb` — Google Colab; clones the repo, stores
  data and checkpoints under `/content/drive/MyDrive/weight-sparsity/`.
* `notebooks/vastai_bottleneck.ipynb` — vast.ai; paths under `/workspace/`, and
  it queues the runs back-to-back detached.

`docs/vastai-agent-guide.md` is the operational guide for the rented-box
workflow (queues, TensorBoard mirrors, Drive backups, checkpoint policy).

## Layout

```
src/wsparse/
  config.py              dataclass configs, YAML (_base_) composition, CLI overrides, legacy migration
  model.py               RMSNorm transformer, learned or rotary positions, no biases
  tokenizer.py           GPT-Neo tokenizer, or a small BPE trained on TinyStories
  data.py                dataset preparation + uint16 memmap batching
  optim.py               AdamW param groups (decay / nodecay) + LR schedule
  decouple.py            magnitude-direction decoupling: md_init_, decoupled optimizer
  train.py               training loop, DDP, evaluation, checkpointing
  interventions.py       counterfactual gradient captures on a loaded checkpoint
  utils.py               device/dtype, seeding, JSONL + TensorBoard + wandb logging
  bottleneck/
    lapsum.py            Laplace CDF, closed-form + reference barrier, exact VJP
    rblapsum.py          rank-boundary kernel, the four boundary-gradient modes, soft forward
    gate.py              hard TopK forward / surrogate backward
    module.py            SparseTopKBottleneck (dense in_proj / gate / out_proj [/ post-norm])
    controller.py        layer selection, placement, diagnostics aggregation
configs/                 composable YAML configs
scripts/                 training queues, TensorBoard mirrors, Drive backups, figure scripts
analysis/                offline probes + the streamlit score explorer
tools/repro_check.py     bit-exactness oracle across code versions
docs/                    the research notes (.tex / .md) behind each result
tests/                   pytest suite
```

## Tests

```bash
pip install pytest && pytest -q
WSPARSE_DDP_TEST=1 pytest -q tests/test_train_ddp.py   # spawns torchrun; run on a box
```

All on synthetic data, no downloads.

* `tests/test_bottleneck.py` — exact-`K` forward (counted from the mask, since a
  selected activation can itself be zero), TopK vs AbsTopK selection and sign
  preservation, the Top-(K+J) gradient support, the closed-form barrier against
  bisection across temperatures/scales/translations/ties, scale and translation
  invariance, the VJP against barrier-re-solved finite differences, placement
  and post-norm wiring, and a sweep of adversarial score geometries (heavy
  tails, 10⁶ scale, offsets, tied scores, bimodal clusters, one 10⁷ spike) for
  NaNs and budget drift.
* `tests/test_rblapsum.py`, `test_rblapsum_sf.py`, `test_rblapsum_perm.py` — the
  rank boundary and floor, the four boundary-gradient modes (including
  `through_rank` against autograd and the zero-sum property of
  `through_rank_kappa`), the soft forward and its value-gradient modes, and the
  permutation ablation's invariants (`ρ=0` bitwise identity, matched marginals).
* `tests/test_laplace_policy.py` — the sampled-support gate (exactly `K`
  inside the clean pool, original values, clean eval, explicit stochastic
  eval), the density's detach rules, the two-candidate estimator against the
  analytic Laplace-difference derivative and a two-gate chain against an
  enumerated expected loss (Monte Carlo, tolerances from the measured standard
  error), schedule / width / rescaling, the sequence-weighted loss, the EMA
  baseline, paired validation, RNG + baseline checkpoint continuation, the
  configuration contract, and an end-to-end training run with resume.
* `tests/test_model.py`, `test_config.py` — architecture, init and logit scaling;
  config composition, CLI overrides, the shipped configs and the legacy migration.
* `tests/test_decouple.py`, `test_md_init.py` — the decoupled optimizer's
  invariants (directions stay on their spheres) and the init-only ablation.
* `tests/test_train.py`, `test_train_ddp.py`, `test_interventions.py` — short
  end-to-end runs, checkpoint round-trips, the data stream, the stop hook, the
  2-process gloo run, and the intervention captures.

## Extending

A new surrogate is a new `surrogate_mode`: add the backward as an
`autograd.Function` next to `lapsum.py` / `rblapsum.py`, dispatch to it in
`AdaptiveLapSumTopKGate.forward`, and add the mode to
`ActivationBottleneckConfig.surrogate_mode`'s validation. The forward stays hard
TopK unless the mode deliberately changes it (`rblapsum_sf` is the one that
does), and the gate's contract is what makes the comparison meaningful: two
modes at the same `(K, J, N, seed)` have a bit-identical forward pass, so any
difference in the curves is a difference in the gradient.

Anything that changes numerics for an existing mode should be checked with
`tools/repro_check.py` against the previous commit before it lands.
