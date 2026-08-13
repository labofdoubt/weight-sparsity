# weight-sparsity

Training small language models (up to ~150M parameters) on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with
handles for two families of **differentiable weight sparsity**:

| method | mask | learnable | paper |
| --- | --- | --- | --- |
| `ltp` | `m = σ(β · (w² − τ))` | one scalar threshold `τ` per layer | [Learned Threshold Pruning](https://arxiv.org/abs/2003.00075) (Azarian et al.) |
| `cs` | `m = σ(β · s)` | a free gate `s` per weight element | [Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427) (Savarese et al.) |

In both cases the forward pass uses `v = w ⊙ m`, the smooth L0 of a layer is
`Σ σ(β·z)`, and `β` (the *inverse* temperature) is raised during training by a
schedule so the sigmoid anneals towards a step function.

Two deviations from the papers, both deliberate:

* LTP's per-layer temperature `T_l = T₀·σ²(|w|)` (eq. 15) is **not** used —
  `β` comes purely from the schedule, identically for both methods.
* Alongside the plain L0 penalty there is an optional **target-density**
  objective that steers each layer towards a prescribed density instead of
  just pushing L0 down.

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
python -m wsparse.train --config configs/ltp_target.yaml   # LTP, 10% target density
python -m wsparse.train --config configs/ltp_150m.yaml     # ~152M parameters

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

Everything is also written to `runs/<run_name>/metrics.jsonl` (and to
Weights & Biases if `train.wandb_project` is set):

| metric | meaning |
| --- | --- |
| `train/ce`, `train/loss` | cross-entropy, and CE + sparsity penalties |
| `sparsity/beta` | current inverse temperature |
| `sparsity/density_soft` | `Σ m / N` — density from the smooth L0 |
| `sparsity/density_hard` | `Σ 1[z > 0] / N` — density after hard pruning |
| `sparsity/transition_frac` | fraction of weights with `|β·z| < 4`, i.e. still inside the sigmoid's transition band. When this hits 0 the mask has frozen and `τ`/`s` stop learning (the stall LTP §3.2 warns about) |
| `sparsity/threshold_mean` / `s_mean` | mean `τ` (LTP) or mean `s` (CS) |
| `sparsity/l0_penalty`, `sparsity/target_penalty` | the two sparsity loss terms |
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
| `method` | `ltp` or `cs` |
| `targets` | `[mlp]` by default; `[mlp, attn]` also masks `qkv` and the attention output projection |
| `beta_schedule` | `constant`, `linear`, `exponential`, `cosine`, `polynomial` |
| `beta_start`, `beta_end` | endpoints of the anneal |
| `beta_warmup_steps` | hold `β` at `beta_start` for this many steps (train dense first) |
| `beta_anneal_steps` | `null` → anneal over everything left after the warmup |
| `beta_power` | exponent for `polynomial` |
| `mask_lr` | learning rate for `τ` (LTP) / `s` (CS); their own AdamW group, no weight decay, no LR schedule |
| `threshold_init` | LTP: initial `τ` |
| `s_init` | CS: initial gate value — the paper's main sparsity knob (−0.3 … 0.3) |
| `grad_through_mask` | LTP only. `false` (paper eq. 14) makes `∂v/∂w = m`, i.e. the sigmoid is treated as constant w.r.t. `w`, while `τ` keeps its full gradient; `true` backpropagates through `w²` as well |
| `l0_coef`, `l0_normalize` | smooth-L0 penalty `l0_coef · Σ_l L0_l`, divided by the total maskable count when normalised (so `l0_coef` is O(1) instead of O(1e-8)) |
| `target_density`, `target_density_coef`, `target_density_overrides` | target-density objective, below |
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

If `sparsity/transition_frac` collapses to 0 early, `β` is rising too fast (or
`mask_lr` is too small to keep up) and the mask will freeze before it has
sorted the weights.

## Notebooks

* `notebooks/colab_tinystories.ipynb` — Google Colab; clones the repo, stores
  data/checkpoints under `/content/drive/MyDrive/weight-sparsity/`.
* `notebooks/vastai_tinystories.ipynb` — vast.ai; paths under `/workspace/`
  with an optional persistent-volume location.

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
  sparsity/
    schedules.py         inverse-temperature schedules
    masks.py             LTPLinear, CSLinear, hard-mask evaluation
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

## Extending

Attention sparsity is already implemented — `sparsity.targets: [mlp, attn]`.
Adding a third masking method means subclassing `SparseLinear` with a `logits()`
and `mask_parameters()`, then registering it in `make_sparse_linear`.
