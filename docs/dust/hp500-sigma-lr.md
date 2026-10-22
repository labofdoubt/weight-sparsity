# DUST on the K=32 code-carried model: the 500-step sigma x dust_lr_scale grid

2026-10-22. Single runs, seed 1337; the encoder arm and the baseline on new_york
(8x RTX 5090), the backprop arm on georgia (8x A100).

**Setup.** The model is the K=32 code-carried hard Top-32 model of
`docs/kj-vs-hard-topk-code-residual.tex` (`configs/cr_kj/ma_cr_hard_k32_pnorm.yaml`:
8 blocks of width 768, abs-TopK over N=1536 at `residual_out` of every block, the
code carried between blocks, post-norm, magnitude-direction decoupling, batch
96 x 512 tokens, 20k-step cosine schedule with 500 warmup steps). Every DUST run
(`configs/cr_dust/hp500/`, `docs/dust-topk.md`) adds the DUST bottleneck
optimization with the J=480 pool, population 128 (16 draws per bottleneck),
schedule [1, 1] (every other step a DUST step), gamma 0, grad scale 1, a
separate optimizer for the DUST steps sharing the MD parameterization, and no
gradient clipping; the grid is `dust_target` in {encoder, backprop} x
`dust_sigma` in {0.03, 0.1, 0.3, 0.9} x `dust_lr_scale` in {1, 10, 30, 100, 300}.
Each run trains 500 steps of the unchanged 20k-step schedule, so the window lies
inside the LR warmup (1.2e-6 to 6e-4; the DUST steps use that value times
`dust_lr_scale`), with validation every 50 steps on 40 batches of 24 windows.
The baseline is the same model trained with the same settings and no DUST
steps (`ny_hp500_hard_k32`); the original madrid run of the model (gradient
clipping at 1.0) is quoted for reference.

## Encoder target

![encoder grid](figures/hp500_val_encoder.png)

Validation CE at step 500:

| sigma \ dust_lr_scale | 1 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| 0.03 | 3.0220 | 3.2500 | 4.0055 | 4.8971 | 5.2906 |
| 0.1 | 3.0096 | 3.1989 | 3.8673 | 5.0334 | 5.5611 |
| 0.3 | 3.0101 | 3.2891 | 4.5535 | 4.9989 | 5.3168 |
| 0.9 | **2.9957** | 3.5009 | 4.2972 | 4.9715 | 5.5825 |
| hard Top-32 baseline (no DUST steps, same settings) | 2.6875 | | | | |
| hard Top-32, the original madrid run (clipping 1.0) | 2.6846 | | | | |

## Backprop target

![backprop grid](figures/hp500_val_backprop.png)

Validation CE at step 500:

| sigma \ dust_lr_scale | 1 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| 0.03 | 3.1246 | 4.6930 | 6.2055 | 13.0487 | 9.7345 |
| 0.1 | 3.1169 | 5.3437 | 6.0478 | 8.8164 | 9.9774 |
| 0.3 | **3.0794** | 5.9428 | 6.2382 | 11.9248 | 9.6194 |
| 0.9 | 3.1646 | 5.8732 | 5.5386 | 5.7687 | 9.2661 |
| hard Top-32 baseline (no DUST steps, same settings) | 2.6875 | | | | |
| hard Top-32, the original madrid run (clipping 1.0) | 2.6846 | | | | |

## Results

No cell reaches the baseline. In both arms the validation CE at step 500 rises
monotonically with `dust_lr_scale`, and the multiplier 1 is the best value at
every sigma: 2.996 to 3.022 in the encoder arm, 0.31 to 0.33 nats above the
baseline, and 3.08 to 3.16 in the backprop arm. The dependence on sigma is weak
at the multiplier 1 (a spread of 0.03 nats in the encoder arm and 0.09 in the
backprop arm) and does not change the ordering by the multiplier anywhere.

The two targets differ in how the larger multipliers fail. In the encoder arm
the curves keep descending at every setting, only more slowly and from a higher
level as the multiplier grows; at 100 and 300 they plateau near 5 to 5.6 nats
within the window. In the backprop arm, where the estimate is backpropagated to
every parameter before the gate inputs, every multiplier from 10 on stalls near
5 to 6 nats, and 100 and 300 diverge at every sigma (final values of 5.8 to 13
nats), with the pre-activation scale growing by orders of magnitude within the
first tens of steps.

Within the first 500 steps the DUST steps therefore cost against hard steps at
every setting of the grid, least when the DUST learning rate is the schedule's
own, and the encoder target is less harmful than the backprop target at every
sigma and multiplier.

## Sigma sweep at population 1024

Pending: the eight runs (`dust_lr_scale` 1, population 1024 = 128 draws per
bottleneck, both targets, the four sigma values, otherwise the grid's settings)
are training on new_york; their two figures and table will be added here.

## Runs on Drive

Encoder arm, new_york (`gdrive:weight-sparsity/runs_new_york/`):
`ny_hp500_enc_sg0p03_lr1`, `ny_hp500_enc_sg0p03_lr10`, `ny_hp500_enc_sg0p03_lr30`,
`ny_hp500_enc_sg0p03_lr100`, `ny_hp500_enc_sg0p03_lr300`,
`ny_hp500_enc_sg0p1_lr1`, `ny_hp500_enc_sg0p1_lr10`, `ny_hp500_enc_sg0p1_lr30`,
`ny_hp500_enc_sg0p1_lr100`, `ny_hp500_enc_sg0p1_lr300`,
`ny_hp500_enc_sg0p3_lr1`, `ny_hp500_enc_sg0p3_lr10`, `ny_hp500_enc_sg0p3_lr30`,
`ny_hp500_enc_sg0p3_lr100`, `ny_hp500_enc_sg0p3_lr300`,
`ny_hp500_enc_sg0p9_lr1`, `ny_hp500_enc_sg0p9_lr10`, `ny_hp500_enc_sg0p9_lr30`,
`ny_hp500_enc_sg0p9_lr100`, `ny_hp500_enc_sg0p9_lr300`.

Baseline, new_york: `gdrive:weight-sparsity/runs_new_york/ny_hp500_hard_k32`.

Backprop arm, georgia (`gdrive:weight-sparsity/runs_georgia/`):
`ga_hp500_bp_sg0p03_lr1`, `ga_hp500_bp_sg0p03_lr10`, `ga_hp500_bp_sg0p03_lr30`,
`ga_hp500_bp_sg0p03_lr100`, `ga_hp500_bp_sg0p03_lr300`,
`ga_hp500_bp_sg0p1_lr1`, `ga_hp500_bp_sg0p1_lr10`, `ga_hp500_bp_sg0p1_lr30`,
`ga_hp500_bp_sg0p1_lr100`, `ga_hp500_bp_sg0p1_lr300`,
`ga_hp500_bp_sg0p3_lr1`, `ga_hp500_bp_sg0p3_lr10`, `ga_hp500_bp_sg0p3_lr30`,
`ga_hp500_bp_sg0p3_lr100`, `ga_hp500_bp_sg0p3_lr300`,
`ga_hp500_bp_sg0p9_lr1`, `ga_hp500_bp_sg0p9_lr10`, `ga_hp500_bp_sg0p9_lr30`,
`ga_hp500_bp_sg0p9_lr100`, `ga_hp500_bp_sg0p9_lr300`.

Original hard Top-32 run (reference value only):
`gdrive:weight-sparsity/runs_madrid/ma_cr_hard_k32_pnorm`.

Figures and tables: `scripts/plot_dust_hp500.py` over the `metrics.jsonl` of
these runs.
