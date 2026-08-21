# weight-sparsity

Training small language models (up to ~150M parameters) on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with
handles for three families of **differentiable weight sparsity**:

| method | mask | learnable | paper |
| --- | --- | --- | --- |
| `ltp` | `m = σ(β · (w² − τ))` | one scalar threshold `τ` per layer | [Learned Threshold Pruning](https://arxiv.org/abs/2003.00075) (Azarian et al.) |
| `cs` | `m = σ(β · s)` | a free gate `s` per weight element | [Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427) (Savarese et al.) |
| `topk` | `m = 1[(i,j) ∈ TopK(s)] · σ(β · s)` | a score `s` per weight element | — ([below](#topk--soft-gate)) |

The forward pass is always `v = w ⊙ m`, and `β` (the *inverse* temperature) is
raised during training by a schedule so the sigmoid anneals towards a step
function. For `ltp`/`cs` the smooth L0 of a layer is `Σ σ(β·z)` and the density
is whatever the penalties drive it to; `topk` instead fixes the forward density
at `k` per group and gives the backward pass a **wider support** than the
forward one.

A separate experiment family, **activation sparsity**, lives alongside these:
`activation_bottleneck` inserts a hard TopK/AbsTopK bottleneck in front of
selected MLPs with a LapSum Top-(K+J) surrogate gradient — see
[Activation bottleneck](#activation-bottleneck). Its projections are dense and
the two families are mutually exclusive.

Three deviations from the papers, all deliberate:

* LTP's per-layer temperature `T_l = T₀·σ²(|w|)` (eq. 15) is **not** used —
  `β` comes purely from the schedule, identically for all three methods.
* Alongside the plain L0 penalty there is an optional **target-density**
  objective that steers each layer towards a prescribed density instead of
  just pushing L0 down.
* `topk` defines its backward pass by hand rather than differentiating the
  forward one, so that inactive candidate weights can still learn.

## Install

```bash
git clone https://github.com/labofdoubt/weight-sparsity.git
cd weight-sparsity
pip install -e .[data]          # torch, numpy, pyyaml, datasets, transformers, tokenizers
```

## Quick start

```bash
# 1. download + tokenise TinyStories into data/tinystories/{train,val}.bin
python -m wsparse.data --config configs/dense.yaml

# 2. train
python -m wsparse.train --config configs/dense.yaml        # dense baseline
python -m wsparse.train --config configs/ltp.yaml          # LTP, L0 penalty
python -m wsparse.train --config configs/cs.yaml           # Continuous Sparsification
python -m wsparse.train --config configs/topk.yaml         # TopK + soft gate, 10% density
python -m wsparse.train --config configs/topk_soft_l0.yaml # ... plus the soft-L0 penalty
python -m wsparse.train --config configs/ltp_target.yaml   # LTP, 10% target density
python -m wsparse.train --config configs/ltp_150m.yaml     # ~152M parameters

# activation sparsity (a separate experiment family)
python -m wsparse.train --config configs/bn_dense.yaml     # 81.7M dense baseline
python -m wsparse.train --config configs/bn_hard.yaml      # + hard TopK bottleneck
python -m wsparse.train --config configs/bn_lapsum.yaml    # + LapSum surrogate gradient

# 3. inspect / sample
python scripts/model_summary.py --config configs/ltp_150m.yaml
python scripts/generate.py --ckpt runs/ltp_small/latest.pt --hard
```

Any field can be overridden from the command line:

```bash
python -m wsparse.train --config configs/ltp.yaml \
    --model.n_layers=16 --train.lr=3e-4 \
    --sparsity.beta_schedule=cosine --sparsity.beta_end=1e5 \
    --sparsity.target_density=0.05 --sparsity.target_density_coef=1.0
```

Configs compose through a `_base_` key (see `configs/*.yaml`); fragments live in
`configs/model`, `configs/train` and `configs/sparsity`.

## What gets printed during training

```
step    200/20000 | loss 3.9214 | ce 3.8714 | ppl   48.02 | lr 6.00e-04 | beta 1.58e+03 \
 | dens_soft 0.8123 | dens_hard 0.7904 | trans 0.0421 | threshold_mean 3.1e-05 \
 | gnorm 0.51 | 41.2K tok/s | 124 ms/step
step    500 | val ce 3.7410 | val ppl 42.15 | hard ce 3.7502 | hard ppl 42.54 | density 0.7801
```

`topk` adds the mean gate over the selected weights and the TopK churn:

```
[train] topk: k=1024 j=1024 per group (groups=tensor), forward density 0.2500, w grad on topk+j
step      8/20000 | ... | beta 7.66 | dens_soft 0.2274 | dens_hard 0.2426 | trans 0.5176 \
 | gate 0.909 | turn 0.0032 | gnorm 0.50 | ...
```

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
| `train/ce`, `train/loss` | cross-entropy, and CE + sparsity penalties |
| `sparsity/beta` | current inverse temperature |
| `sparsity/density_soft` | `Σ m / N` — density from the smooth L0 |
| `sparsity/density_hard` | `Σ 1[z > 0] / N` — density after hard pruning |
| `sparsity/transition_frac` | fraction of weights with `|β·z| < 4`, i.e. still inside the sigmoid's transition band. When this hits 0 the mask has frozen and `τ`/`s` stop learning (the stall LTP §3.2 warns about). For `topk` it is measured over the Top-(k+j) support, since nothing else gets a gradient anyway |
| `sparsity/threshold_mean` / `s_mean` | mean `τ` (LTP) or mean `s` (CS) |
| `sparsity/density_topk` | `topk` only: `\|A\|/N`, the hard FLOP budget. `density_soft` can only sit below it |
| `sparsity/gate_mean_topk` / `gate_mean_explore` | `topk` only: mean gate `σ(β·s)` over the selected weights, and over the `j` candidates |
| `sparsity/turnover` | `topk` only: fraction of TopK that changed at the last re-selection. **If this is 0 from early on, `j` is buying you nothing** — the exploratory candidates never overtake an incumbent |
| `sparsity/l0_penalty`, `sparsity/target_penalty`, `sparsity/soft_l0_penalty` | the sparsity loss terms |
| `sparsity/mask_grad_norm` | gradient norm of `τ`/`s`, clipped separately from the weights (`∂L/∂τ` sums over a whole layer, so a shared global clip would squash the weight gradients) |
| `val/ce`, `val_hard/ce` | validation loss with soft masks and with binary masks |
| `layer_*` | per-layer hard density, logged at every validation |

The soft/hard gap is the thing to watch: it is what tells you whether the
annealing has gone far enough for the pruned network to be the network you
actually trained.

## Configuration

### `model`

| field | default | notes |
| --- | --- | --- |
| `n_layers`, `d_model`, `n_heads` | 12 / 768 / 12 | |
| `mlp_ratio` | 4.0 | `d_mlp = round(mlp_ratio · d_model)`, rounded up to a multiple of 8 |
| `mlp_activation` | `gelu` | `gelu`, `relu`, `silu`, `swiglu` |
| `max_seq_len` | 512 | learnable positional embedding table size |
| `bias` | `false` | no biases anywhere by default |
| `norm_eps` | 1e-6 | RMSNorm (pre-norm blocks, plus a final norm) |
| `tie_embeddings` | `true` | shares `lm_head` with the token embedding |
| `init_scheme` | `fixed_std` | `fixed_std`: `w ~ N(0, init_std²)`; `fan_in`: `w ~ N(0, init_gain²/fan_in)` |
| `init_std`, `init_gain` | 0.02, 1.0 | weight standard deviation / fan-in gain |
| `init_std_embedding`, `init_std_pos` | `null` | default to `init_std` |
| `init_scale_residual` | `true` | scales every residual output projection by `1/√(2·n_layers)` |
| `dropout`, `attn_dropout` | 0.0 | |

Sizes: `configs/model/{tiny,small,medium,large}.yaml` →
3M / 25M / 85M / 113M non-embedding parameters (16M / 51M / 124M / **152M**
total with the 50257-token GPT-Neo vocabulary).

### `data`

`tokenizer: gpt_neo` (default) uses the GPT-Neo/GPT-2 byte-level BPE, i.e. the
tokenizer the original TinyStories models were trained with. `tokenizer: bpe`
instead trains a small byte-level BPE on TinyStories itself
(`bpe_vocab_size`, default 8192), which shrinks the embedding matrix a lot and
puts more of the parameter budget into the transformer body.

### `train`

`optimizer` (AdamW), `lr`, `betas`, `eps`, `weight_decay`, `grad_clip`,
`lr_schedule` (`cosine`/`linear`/`constant`), `warmup_steps`, `min_lr_ratio`,
`batch_size` (sequences per optimiser step), `micro_batch_size` (gradient
accumulation = `batch_size / micro_batch_size`), `max_steps`,
`log_every_steps`, `validate_every_steps`, `val_batches`,
`checkpoint_every_steps`, `keep_last_checkpoints`, `sample_every_steps`,
`seed`, `device`, `dtype`, `compile`, `out_dir`, `run_name`, `resume`,
`wandb_project`.

Weight decay is applied only to ≥2D weights; RMSNorm gains, biases and all
sparsity parameters are excluded.

### `sparsity`

| field | notes |
| --- | --- |
| `enabled` | master switch |
| `method` | `ltp`, `cs` or `topk` |
| `targets` | `[mlp]` by default; `[mlp, attn]` also masks `qkv` and the attention output projection |
| `beta_schedule` | `constant`, `linear`, `exponential`, `cosine`, `polynomial` |
| `beta_start`, `beta_end` | endpoints of the anneal |
| `beta_warmup_steps` | hold `β` at `beta_start` for this many steps (train dense first) |
| `beta_anneal_steps` | `null` → anneal over everything left after the warmup |
| `beta_power` | exponent for `polynomial` |
| `inverse_temperature` | alias for `β`: on its own it pins a **constant** `β(t) = β₀`; combined with a schedule it sets `beta_start` |
| `inverse_temperature_schedule` | alias that selects `beta_schedule`, i.e. `β(t)` from `inverse_temperature` up to `beta_end` |
| `mask_lr` | learning rate for `τ` (LTP) / `s` (CS, TopK); their own AdamW group, no weight decay, no LR schedule |
| `mask_lr_mult` | instead pin that lr to `mask_lr_mult · train.lr`, in which case it *does* follow the LR schedule |
| `mask_grad_clip` | clip norm for the sparsity parameters (`null` reuses `train.grad_clip`); they are always clipped separately from the weights |
| `threshold_init` | LTP: initial `τ` |
| `s_init` | CS: initial gate value — the paper's main sparsity knob (−0.3 … 0.3). TopK: the scale of the score init |
| `s_init_mode` | TopK: `magnitude` (default in the shipped config), `constant`, `uniform`, `normal` |
| `grad_through_mask` | LTP only. `false` (paper eq. 14) makes `∂v/∂w = m`, i.e. the sigmoid is treated as constant w.r.t. `w`, while `τ` keeps its full gradient; `true` backpropagates through `w²` as well |
| `k`, `j` | TopK: active weights per group, and extra exploratory positions per group. A fraction (`< 1`) of the group, or an absolute count (`≥ 1`) |
| `topk_groups`, `topk_block_size` | TopK: `tensor` (one global TopK), `row` (one per output row) or `block` (one per run of `topk_block_size` weights, i.e. `k`:`block_size` structured sparsity) |
| `w_grad_support` | TopK: `topk_j` gives `w` gradients on the whole Top-(k+j) support, `topk` restricts them to TopK (`s` always explores) |
| `topk_track_turnover` | TopK: log how much of TopK changes per step; costs one extra bool tensor per masked layer |
| `soft_l0_enabled`, `soft_l0_lambda_topk`, `soft_l0_lambda_explore` | TopK: soft-L0 penalty over the Top-(k+j) support, below |
| `l0_coef`, `l0_normalize` | smooth-L0 penalty `l0_coef · Σ_l L0_l`, divided by the total maskable count when normalised (so `l0_coef` is O(1) instead of O(1e-8)) |
| `target_density`, `target_density_coef`, `target_density_overrides` | target-density objective, below. Rejected for `topk`, whose density is `k` by construction |
| `eval_hard_mask` | also run validation with binary masks |

#### The two sparsity objectives

```
loss = CE
     + l0_coef            · Σ_l L0_l                       (/ N if l0_normalize)
     + target_density_coef · mean_l (L0_l / D_l − 1)²
```

`D_l = target_density_l · numel_l`, so you specify a **dense fraction in
(0, 1]** and the actual weight count is derived from the architecture.
`target_density` sets it for every masked layer; `target_density_overrides`
overrides it per layer using `fnmatch` patterns against the module name
(later patterns win):

```yaml
sparsity:
  target_density: 0.1
  target_density_coef: 1.0
  target_density_overrides:
    "blocks.0.mlp.*": 0.4      # keep the first block denser
    "*.mlp.fc2": 0.15
```

Both coefficients can be non-zero; setting one to 0 turns that term off.

#### TopK + soft gate

`method: topk` keeps a score `s` per weight alongside `w` and gates it with
`p = σ(β·s)`. Two index sets are derived from `s` alone (equivalently from `p`,
the sigmoid being monotone):

```
A = TopK_k(s)        the forward support,  |A| = k per group
B = Top_(k+j)(s)     the backward support, A ⊆ B
```

The forward pass keeps only `A`, attenuated by the gate — everything else is
*exactly* zero, so `k` is a hard FLOP budget rather than something a penalty
has to negotiate:

```
w̃ = M_A ⊙ w ⊙ σ(β·s)
```

The backward pass is then **defined**, not derived — hard TopK is never
differentiated through. With `G = ∂L/∂w̃` (nonzero even where `w̃ = 0`, since it
is the gradient w.r.t. the effective weight and not the layer output):

```
∂L/∂w = M_B ⊙ p ⊙ G
∂L/∂s = M_B ⊙ [G ⊙ w + Λ] ⊙ β·p·(1−p)
```

So the `j` positions in `B \ A` cost nothing in the forward pass but still
accumulate gradients for **both** `w` and `s`, and can climb into TopK on a
later step. Positions outside `B` get exactly zero. This is one forward and one
backward pass per step — the widened support is what the custom
`torch.autograd.Function` in `sparsity/topk.py` buys, and `sparsity/turnover`
in the logs is how you check the exploration is actually doing something.

`train.compile: true` is fine here: the selection itself is hidden from Dynamo
(otherwise its cache key — the version counter of `s` — would force a recompile
on every optimiser step), while the matmuls around it still compile, and the
compiled gradients match eager exactly.

`Λ` is the optional soft-L0 penalty, computed over `B` only and never over the
dense tensor:

```
L_sparse = λ_K · Σ_{(i,j) ∈ A} p_ij  +  λ_J · Σ_{(i,j) ∈ B\A} p_ij
```

It does not change the fact that the forward pass uses exactly `k` positions;
what it buys is *soft* sparsity **inside** the TopK support, since a selected
gate can be driven towards 0 while still occupying a slot in the budget. Watch
`density_soft` fall below `density_topk` for that. Unlike `l0_coef` these
lambdas are **not** normalised — they are per-weight coefficients, so they live
on the `l0_coef / maskable_count` scale (`~1e-7`, not `~0.05`).

```yaml
sparsity:
  method: topk
  k: 0.1                    # 10% of every masked tensor is active
  j: 0.05                   # another 5% explores
  topk_groups: block        # with topk_block_size: 4 and k: 2 -> 2:4 sparsity
  inverse_temperature: 4.0
  inverse_temperature_schedule: exponential
  beta_end: 200.0
  soft_l0_enabled: true
  soft_l0_lambda_topk: 1.0e-7
```

`s_init_mode: magnitude` (the shipped default) initialises `s` from `|w|`,
scaled so that its per-group standard deviation is about `s_init` and centred
on the selection boundary: the initial TopK is then the top-`k` weights by
magnitude, and `s > 0` holds on exactly that set, so TopK, the hard mask and
the soft gate all agree at step 0. `constant` leaves the selection to index
order and is only useful for tests.

#### Choosing `β`

`β` multiplies `z`, and the scale of `z` differs enormously between the two
methods — the defaults reflect that, and it is the one thing to re-tune if you
change the initialisation:

* **LTP**: `z = w² − τ`. With `init_std = 0.02` a typical `w²` is ~4e-4, so `β`
  has to exceed ~2.5e3 before the mask stops being ~0.5 everywhere. Default:
  exponential `1e4 → 1e6`, which starts the model essentially dense
  (`σ(1e4·4e-4) ≈ 0.98`) and sparsifies as `τ` rises and the sigmoid sharpens.
  `τ` lives on that same ~1e-4 scale, and AdamW moves a parameter by roughly
  `lr` per step regardless of gradient size, hence the tiny default
  `mask_lr = 1e-6`.
* **CS**: `z = s`, order 0.05–1. Default: exponential `1 → 200` and
  `mask_lr = 1e-3`, as in the paper.
* **TopK**: `z = s` as for CS, but `s` is initialised with a per-group standard
  deviation of `s_init`. Default: `inverse_temperature: 4` annealed to 200,
  which starts with the gates spread across most of (0, 1). Note the trap in
  `∂L/∂s ∝ β·σ(β·s)·(1−σ(β·s))`: raising `β` sharpens the gate towards
  `1[s > 0]`, but it also concentrates that factor into an ever-narrower spike
  around `s = 0`, so past `β ≈ 1e3` the scores of the *already-decided* weights
  stop moving and TopK freezes. `inverse_temperature` on its own pins `β`
  constant, which is the clean way to take the schedule out of the picture
  while you tune `k` and `j`.

If `sparsity/transition_frac` collapses to 0 early, `β` is rising too fast (or
`mask_lr` is too small to keep up) and the mask will freeze before it has
sorted the weights. For `topk`, `sparsity/turnover` going to 0 says the same
thing about the selection.

## Activation bottleneck

`activation_bottleneck` is an **activation**-sparsity experiment, unrelated to
the weight sparsity above. It inserts a dense-in / sparse-gate / dense-out
bottleneck on the tensor that feeds each selected MLP:

```
x_mlp ──▶ W_in ──▶ TopK / AbsTopK  (exactly K of N) ──▶ W_out ──▶ MLP
```

In `Block.forward` that is `self.norm2(x)` — the pre-RMSNorm MLP input, **not**
the residual stream, which stays dense. `W_in` and `W_out` are ordinary dense
`nn.Linear` layers trained by the model's ordinary objective: no reconstruction
loss, no weight mask, no pruning. Enabling `sparsity` and
`activation_bottleneck` together is a config error, not a silent combination.

### Forward: exact hard TopK

With ranking score `r = a` (`topk`) or `r = |a|` (`abs_topk`):

```
â_i = a_i · 1[i ∈ TopK_K(r)]
```

Selection is by magnitude in `abs_topk`, but the **signed** activation is
forwarded — never `|a_i|`.

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
the support. That falls out of the same `m_hard + λ(p − stopgrad(p))` mask
applied to a *separate* value tensor, so it reuses the existing VJP rather than
re-deriving the Jacobian. Costs one extra `d_model × n_features` projection per
layer (39.4M vs 26.2M at `d_model=640, N=2048`). The forward pass is exactly `K`-sparse no matter what
`J`, `n_eff`, the metric or the boundary mode are set to; the LapSum
probabilities below are never used numerically in the forward pass. There is no
ReLU before the TopK.

### Backward: LapSum over Top-(K+J)

One `torch.topk(K+J, sorted=True)` per call produces the sorted candidate pool
that the hard mask, the temperature solve, the barrier solve, the probabilities,
the backward and the diagnostics all share. Its first `K` entries are the active
set `A`; the remaining `J` are inactive candidates. Everything outside
Top-(K+J) receives exactly zero gradient from this module.

Over that pool, with the Laplace CDF `F(z) = ½eᶻ` for `z ≤ 0` and `1 − ½e⁻ᶻ`
otherwise, the soft mask is `p_i = F((r_i − b)/t)` with the barrier `b` fixed by
`Σᵢ pᵢ = K`. The mask handed to the layer is

```
m = m_hard + λ · (p − stopgrad(p))
```

so `m == m_hard` numerically while `∂m/∂r == λ·∂p/∂r`. Writing `κᵢ = φᵢ/t`,
`uᵢ = gᵢaᵢ` and `q^budget = κ/Σκ`, the exact fixed-`t` VJP is

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

### Temperature is an adaptive bandwidth

`t` is not a fixed number and not a learned parameter. Score scales drift across
tokens, layers and training, so each row solves for the `t` whose
boundary-exchange weights have a target effective size `n_eff`:

| metric | `N_eff` | behaviour |
| --- | --- | --- |
| `ess` | `1 / Σqᵢ²` | ignores a long tail of small weights |
| `entropy` | `exp(−Σqᵢ log qᵢ)` | credits the tail; needs a colder `t` for the same target |

(`entropy`, not "cross-entropy" — there is no second distribution here.)

Which weights get calibrated is set by two knobs:

| `boundary_mode` | `one_sided_weight_mode` | calibration weights `q` | decouples? |
| --- | --- | --- | --- |
| `outside_only` | `score_softmax` | `softmax(rᵢ/t)`, inactive only | **yes** |
| `outside_only` | `true_gradient` | `κᵢ/Σ_{j>K}κⱼ`, inactive only | no |
| `both_sides` | *(not consulted)* | `κᵢ/Σⱼκⱼ`, all `M` | no |

`score_softmax` is the cheap **approximation**. Because that `q` never mentions
the barrier, the temperature and barrier solves decouple completely: a scalar
root-find over `J` scores, then one closed-form `b`. No barrier solve inside the
temperature loop. It asks *how many inactive features effectively compete to
enter TopK*, defined from inactive score geometry.

It is only *equal* to the normalised LapSum gradient weights when every outside
candidate lies below the barrier, i.e. `r_{K+1} < b`. At finite temperature the
budget-preserving barrier can move above `r_{K+1}`, and then the two differ. The
`true_gradient` and `both_sides` modes calibrate on the actual `κ` instead;
since `κ` depends on `b`, they solve `(b, τ)` jointly by a batched damped Newton
with a 2×2 Jacobian per row, initialised from the cheap `score_softmax` `t₀` and
its closed-form `b₀`. The two exact modes share one solver and differ only in
which indices enter the effective-count equation — a `calibration` slice.

There is no `score_softmax` variant of `both_sides`: the cancellation that
motivates it is a one-sided construction.

**Does the approximation matter?** It is measured every step rather than
assumed. On Gaussian activations with `K=64, J=192`:

| target `n_eff` | `(r_{K+1}−b)/t` | rows with `r_{K+1} > b` | `N_score` | `N_true` | gap |
| --- | --- | --- | --- | --- | --- |
| 2 | −1.51 | 16% | 2.000 | 2.076 | +0.076 |
| 4 | −0.39 | 22% | 4.000 | 4.160 | +0.160 |
| 16 | −0.17 | 18% | 16.000 | 16.128 | +0.128 |
| 64 | −0.27 | 0% | 64.000 | 64.000 | −0.000 |
| 160 | −0.47 | 0% | 160.000 | 160.000 | +0.000 |

So the shortcut is *exact* at large `n_eff` — a warm `t` lifts the barrier clear
of the whole pool — and mildly off (1–4% of the realized count, on a minority of
rows) at small `n_eff`. `frac_above_barrier`, `barrier_gap` and `n_eff_gap` are
logged so this can be checked per run rather than assumed.

All modes solve in `τ = log t`, which guarantees `t > 0` and makes the
multiplicative temperature updates numerically convenient. Scale equivariance
does **not** come from the log: it comes from the calibration equations
depending only on score *differences* divided by `t`, together with
scale-equivariant initialisation and bracketing (both are built from the
candidate span, never from an absolute temperature). The consequence is tested:
`r → cr` gives `t → ct` and `b → cb` with `p`, the normalised calibration
weights and the realized `N_eff` unchanged.

**`K`, `J` and `n_eff` are three independent knobs.** `K` is how many features
are active, `K+J` how many are eligible for gradient, `n_eff` how concentrated
the gradient is *within* that pool. `K = 0.1N, K+J = 0.5N, n_eff = 0.05N` is a
perfectly ordinary configuration. `J` is not `n_eff`.

### Prescribing the temperature instead of solving for it

`surrogate_mode` picks how `t` is obtained:

| mode | `t` | cost |
| --- | --- | --- |
| `lapsum_adaptive` | solved each step so `N_eff` hits its target | one root-find |
| `lapsum_scheduled` | `temperature_schedule` from `_start` to `_end` | no solve |
| `lapsum_fixed` | constant `fixed_temperature`, absolute | no solve |
| `hard` | — no surrogate at all | — |

The scheduled mode reuses the same interpolators as the weight-sparsity `beta`
anneal (`constant`, `linear`, `exponential`, `cosine`, `polynomial`, with
`temperature_warmup_steps` / `temperature_anneal_steps` / `temperature_power`),
now shared in `wsparse/schedules.py`. `exponential` is geometric, i.e. linear in
`log t`, which is the natural interpolation for a temperature. The usual
direction is *falling* — a broad boundary gradient early, a sharp one late,
the mirror image of raising `beta` — but nothing assumes `end < start`.

**`temperature_scale_mode` is the knob that matters here.** `t` lives in raw
score units, and the activation scale drifts across layers and over training, so
an absolute temperature quietly means something different at every point.
`relative` (the default) multiplies the schedule by the per-row standard
deviation of the Top-(K+J) scores, which makes the configured number exactly the
`bottleneck/temperature_rel` that gets logged. Rescaling the activations by 100×
with everything else held fixed:

| `temperature_scale_mode` | `N_eff` at 0.1× | at 1× | at 10× |
| --- | --- | --- | --- |
| `absolute` | 189.2 | 72.8 | 5.9 |
| `relative` | 31.1 | 30.6 | 30.8 |
| *(`lapsum_adaptive`, for reference)* | 16.00 | 16.00 | 16.00 |

So `absolute` is only sensible when you already know the score scale and intend
to pin it; `relative` is the scheduled analogue of the adaptive mode, and the
adaptive mode is the version that holds `N_eff` exactly rather than
approximately. `configs/bn_sched.yaml` and
`configs/bottleneck/baseline_scheduled_temperature.yaml` are the ablations.

### The temperature is a detached bandwidth

> The adaptive temperature is a **detached per-token bandwidth**. The backward is
> the exact LapSum VJP conditional on that fixed bandwidth; it is *not* the
> derivative of the full `t(r)` solver.

Concretely: inspect the scores, choose `t`, detach it, then use the exact LapSum
gradient at that fixed `t`. Nothing differentiates through a Newton or bisection
iteration, and the `N_eff` calibration equation contributes no gradient at all —
including in the joint `true_gradient` and `both_sides` solvers, where `b` and
`t` come out of the same solve but only `b`'s dependence on the scores is
carried into the backward. That dependence, `b = b(r; t)` via `Σpᵢ = K`, is
exactly what the `⟨q^budget, u⟩` term encodes.

This is a deliberate experimental choice, not an approximation forced by the
implementation. `differentiate_temperature: true` is rejected rather than
silently ignored; an implicit gradient through `t(r)` would be a separate mode
with its own tests.

### The closed-form barrier

`Σᵢ F((rᵢ−b)/t) = K` is solved in closed form, not by iteration. On the interval
`r_j ≥ b ≥ r_{j+1}` the budget is `j − ½e^{b/t}A_j + ½e^{−b/t}B_j`, so with
`y = e^{b/t}`:

```
A_j y² + 2(K−j) y − B_j = 0,     A_j = Σ_{i≤j} e^{−rᵢ/t},  B_j = Σ_{i>j} e^{rᵢ/t}
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
  rather than `r_max` is what makes the clamp safe: whenever the span/`t` ratio
  is large enough for it to bite, `Σp = K` forces `b` into the `r_K`/`r_{K+1}`
  gap.
* The `barrier_failures` diagnostic compares against an achievable-precision
  floor, not a flat tolerance. Scores arrive already rounded, so `(rᵢ − b)`
  carries `~eps·|r|` of error that `1/t` amplifies; a flat `1e-6·K` threshold
  reports a failure on every batch of offset activations while the solver is in
  fact exact.

### Config

```yaml
activation_bottleneck:
  enabled: true
  layers: all              # all | even | odd | first:n | last:n | [0, 2, 4]
  placement: pre_mlp
  n_features: 2048         # N
  k: 256                   # K, active in the forward pass
  j: 768                   # J, extra candidates that only receive gradient
  n_eff: 32.0              # target effective boundary participants
  selection_mode: abs_topk         # topk | abs_topk | gated_topk
  effective_count_metric: ess      # ess | entropy
  boundary_mode: outside_only      # outside_only | both_sides
  one_sided_weight_mode: score_softmax  # score_softmax | true_gradient
  surrogate_mode: lapsum_adaptive  # lapsum_adaptive | lapsum_scheduled | lapsum_fixed | hard
  surrogate_grad_scale: 1.0
  fixed_temperature: 1.0           # surrogate_mode: lapsum_fixed only, absolute
  # surrogate_mode: lapsum_scheduled
  temperature_schedule: exponential  # constant | linear | exponential | cosine | polynomial
  temperature_start: 0.5
  temperature_end: 0.02
  temperature_warmup_steps: 0
  temperature_anneal_steps: null   # null -> anneal over the rest of training
  temperature_power: 2.0
  temperature_scale_mode: relative # relative (x per-row score std) | absolute
  temperature_solver_tol: 1.0e-5
  temperature_solver_max_iters: 12
  barrier_solver_tol: 1.0e-6
  solver_dtype: float32
  log_diagnostics: true
  hard_inference: true             # skip the LapSum machinery outside training
```

Static validation covers the obviously impossible: `1 ≤ K < N`, `J ≥ 1`,
`K+J ≤ N`, and `1 < n_eff < J` (one-sided) or `1 < n_eff < K+J` (two-sided).

Those bounds are necessary, not sufficient, and **feasibility is decided
numerically per row**. Two-sided calibration cannot go as low as `N_eff = 1`:
as `t → 0` the two neurons straddling the boundary acquire equal density, so it
bottoms out near 2 (measured: `1.9 < N_eff` at `t/span = 2.5e-3`, against 1 for
one-sided, which drops the `r_K` side). Ties move the floor again. So the
reference solver evaluates `N_eff` at both ends of a scale-relative `log t`
range and reports a per-row status — `target_below_attainable_range`,
`target_above_attainable_range`, `degenerate_scores` — which surfaces as
`bottleneck/status_*`. An infeasible target is never silently returned as a
converged solve: Newton reports failure, the reference runs, and the status is
logged.

(The low end of that scan stops at `span·e⁻⁹` deliberately. Further down the
surrogate is numerically dead — every `|z|` leaves float range, the budget
equation is flat so `b` is only defined up to its plateau, and the weight
softmax collapses onto whichever candidate rounding favours. That reads as
`N_eff → 1` and would make an unreachable target look reachable at a temperature
that transmits no gradient at all.)
`differentiate_temperature` exists but rejects `true` — the solved temperature
is a detached bandwidth choice, and nothing differentiates through a Newton or
bisection iteration. An implicit gradient through `t(r)` would be a separate,
separately-tested mode.

### Experiment matrix

Three matched configs drive the headline comparison, and are what the notebooks
use. They share a model (10 layers, `d_model=640`, **81.7M** parameters dense),
a seed and every training hyper-parameter:

| config | params | forward | backward |
| --- | --- | --- | --- |
| `bn_dense.yaml` | 81.7M | ordinary transformer | ordinary |
| `bn_hard.yaml` | 107.9M | `W_in` → TopK(256 of 2048) → `W_out` | hard mask only |
| `bn_lapsum.yaml` | 107.9M | **identical to `bn_hard`** | LapSum Top-(K+J) surrogate, adaptive `t` |
| `bn_sched.yaml` | 107.9M | **identical to `bn_hard`** | same surrogate, scheduled `t` |

`dense` vs `hard` isolates the cost of the bottleneck; `hard` vs `lapsum`
isolates the surrogate; `sched` vs `lapsum` isolates the adaptive calibration
from simply annealing `t`. All four share an architecture, a parameter count and
a bit-identical forward pass (bar `dense`). The 26.2M gap is the `2·d_model·N` projection
pair on each of the 10 layers, which is why `hard` rather than `dense` is the
control for the gradient question.

`configs/bottleneck/` then covers the full calibration grid plus baselines:

```
{topk,abs_topk}_one_score_{ess,entropy}.yaml   one-sided, cheap score-softmax weights
{topk,abs_topk}_one_true_{ess,entropy}.yaml    one-sided, exact gradient weights
{topk,abs_topk}_two_true_{ess,entropy}.yaml    two-sided, exact gradient weights
baseline_hard_topk.yaml                        hard TopK, ordinary hard-mask backward
baseline_hard_abs_topk.yaml                    hard AbsTopK, ordinary hard-mask backward
baseline_scheduled_temperature.yaml            LapSum, t annealed on a schedule
baseline_fixed_temperature.yaml                LapSum at a fixed absolute t
```

Pairing `one_score` against `one_true` at the same metric and target isolates
exactly what the cheap approximation costs.

Sweep the four scale knobs independently, e.g.

```bash
python -m wsparse.train --config configs/bottleneck.yaml \
    --activation_bottleneck.n_features=4096 --activation_bottleneck.k=256 \
    --activation_bottleneck.j=1792 --activation_bottleneck.n_eff=64 \
    --activation_bottleneck.layers=last:6
```

### Diagnostics

Per bottleneck layer, averaged across layers in the logs: `temperature`,
`temperature_rel` (`t`/std of the candidate scores), `barrier`,
`n_eff_realized`, `n_eff_error`, `barrier_gap` (`r_{K+1} − b`),
`barrier_gap_rel`, `frac_above_barrier`, `n_eff_score`, `n_eff_true_gradient`,
`n_eff_gap`, `status_*`, `budget_residual` (`|Σp − K|`), `score_gap`
(`r_K − r_{K+1}`), `score_span` (`r_K − r_{K+J}`), `feature_dead_frac`,
`feature_usage_entropy`, `feature_usage_max`, `grad_active`,
`grad_inactive`, `grad_rank_bin0..7` (surrogate gradient magnitude by candidate
rank), `temp_iters`, `temp_degenerate`, `temp_unbracketed`, `barrier_failures`,
`newton_iters`, `newton_failed`. The console line carries `t`, `t/std`, `neff`
and `dK`.

The point of the experiment is the gradient-flow behaviour and its failure
modes, so the failure counters are first-class: a row Newton cannot solve falls
back to the slow reference root search and is *counted*, never silently
returned as an invalid temperature.

**`feature_dead_frac` and `feature_usage_entropy` deserve particular attention.**
The characteristic failure of a TopK activation bottleneck is collapse: a subset
of the `N` features wins every token, the rest are never selected, and their
`W_in`/`W_out` columns stop receiving gradient entirely — so the effective width
is far below `N`. The loss, the budget residual and `N_eff` all look perfectly
healthy while that happens, so nothing else logged here would reveal it. Usage
is tracked as a bias-corrected EMA (a uniform-seeded one would take ~460 steps
to decay past the dead threshold, reporting 0% dead throughout the early phase
when collapse is most likely). `feature_usage_entropy` is `exp(H)/N`: 1.0 is
even usage, and it falls to the surviving fraction — on a synthetic collapse
where 64 of 256 features take every slot, it reads 0.250 and `feature_dead_frac`
reads 0.750, both from step 10.

### Cost

Measured on CPU, gate only, 4096 rows at `N=2048, K=256, J=768, n_eff=32`,
forward + backward, relative to a hard-TopK backward:

| variant | ms | vs hard TopK |
| --- | --- | --- |
| hard TopK backward | 82 | 1.00× |
| fixed-temperature LapSum | 245 | 2.99× |
| one-sided ESS | 287 | 3.51× |
| one-sided entropy | 301 | 3.68× |
| two-sided ESS | 473 | 5.79× |
| two-sided entropy | 525 | 6.41× |

End-to-end on a 6-layer `d=384` model with all six layers bottlenecked, the
one-sided path costs about 1.8× a hard-TopK bottleneck and the two-sided about
2.5×; against a dense-MLP-input baseline the whole bottleneck (hard TopK
included) is about 2.1×, since it adds two `d_model × N` projections per layer.
GPU ratios will differ — the solvers are many small reductions — and none of
this is paid at inference, where `hard_inference` skips the machinery entirely.

## Notebooks

Weight sparsity:

* `notebooks/colab_tinystories.ipynb` — Google Colab; clones the repo, stores
  data/checkpoints under `/content/drive/MyDrive/weight-sparsity/`.
* `notebooks/vastai_tinystories.ipynb` — vast.ai; paths under `/workspace/`
  with an optional persistent-volume location.

Activation bottleneck — four matched runs (`SETUP = 'dense' | 'hard' |
'lapsum' | 'sched'`), with a parameter table, shared hyper-parameters and seed, curves for
the adaptive bandwidth and the calibration, and a comparison table:

* `notebooks/colab_bottleneck.ipynb` — Google Colab.
* `notebooks/vastai_bottleneck.ipynb` — vast.ai; also queues all three runs
  back-to-back detached.

## Layout

```
src/wsparse/
  config.py              dataclass configs, YAML (_base_) composition, CLI overrides
  model.py               RMSNorm transformer, learnable pos-emb, no biases
  tokenizer.py           GPT-Neo tokenizer, or a small BPE trained on TinyStories
  data.py                dataset preparation + uint16 memmap batching
  optim.py               AdamW param groups (decay / nodecay / mask) + LR schedule
  train.py               training loop, evaluation, checkpointing
  utils.py               device/dtype, seeding, JSONL + wandb logging
  bottleneck/            activation sparsity (independent of sparsity/)
    lapsum.py            Laplace CDF, closed-form + reference barrier, exact VJP
    temperature.py       one-sided and two-sided adaptive-temperature solvers
    gate.py              hard TopK forward / LapSum Top(K+J) backward
    module.py            SparseTopKBottleneck (dense in_proj / gate / out_proj)
    controller.py        layer selection, diagnostics aggregation
  schedules.py           constant/linear/exponential/cosine/polynomial anneals
  sparsity/
    masks.py             LTPLinear, CSLinear, hard-mask evaluation
    topk.py              TopKSoftGateLinear + its custom autograd Function
    controller.py        layer selection, β broadcast, penalties, statistics
configs/                 composable YAML configs
scripts/                 model_summary.py, generate.py
tests/                   pytest suite (masks, gradients, penalties, training)
```

## Tests

```bash
pip install pytest && pytest -q
```

The suite covers the mask formulas, both LTP gradient paths (analytically,
against hand-derived gradients), the β schedules, the penalty terms and
per-layer density targets, optimiser grouping, and short end-to-end training
runs including checkpoint round-trips — all on synthetic data, no downloads.

`tests/test_bottleneck.py` covers the activation bottleneck: exact-K forward
(counted from the mask, since a selected activation can itself be zero), TopK vs
AbsTopK selection and sign preservation, the Top-(K+J) gradient support, the
closed-form barrier against bisection across temperatures/scales/translations/ties,
both temperature solvers against their references, scale and translation
invariance, the VJP against barrier-re-solved finite differences, and a sweep of
adversarial score geometries (heavy tails, 10⁶ scale, offsets, tied scores,
bimodal clusters, one 10⁷ spike) for NaNs and budget drift.

`tests/test_topk.py` does the same for TopK + soft gate: the forward support,
both gradient formulas and the penalty gradient are checked against the boxed
expressions by hand rather than against autograd — which is the point, since
autograd through the hard TopK would give the *narrow* backward support and
here it must be the wide one.

## Extending

Attention sparsity is already implemented — `sparsity.targets: [mlp, attn]`.
Adding a fourth masking method means subclassing `SparseLinear` with a
`logits()` and `mask_parameters()`, then registering it in
`make_sparse_linear(linear, cfg)`. Override `effective_weight()` (as `topk.py`
does) if the layer needs a backward pass that is not the derivative of its
forward pass, and `extra_penalty()` if it carries its own loss term.
