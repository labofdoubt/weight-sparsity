# The code-residual K+J grid

The 25 cells of docs/kj-vs-hard-topk.tex, each with
`activation_bottleneck.code_residual: true` as the only functional change:

| family | cells | configs |
| --- | --- | --- |
| RBLapSum `through_rank_kappa`, T = 2, no output norm | (K, J) with K, K+J in {32, 64, 128, 256, 512}, J > 0 | `ma_cr_rbk_k<K>_j<J>_t2.yaml` |
| the same at T = 1 with an RMSNorm on each bottleneck output | the same ten (K, J) | `ma_cr_rbk_k<K>_j<J>_pnorm.yaml` |
| hard Top-K' with the post-norm | K' in {32, 64, 128, 256, 512} | `ma_cr_hard_k<K'>_pnorm.yaml` |

Every file is the archived `config.json` of its cell's run (named in the
header, on Drive under `runs_california/` or `runs_korea/`), migrated through
`wsparse.config.config_from_dict` onto the current schema by
`scripts/make_cr_kj_configs.py`, which also checks that nothing but
`train.run_name` and `code_residual` changed in value.  The header of each
file lists the fields the migration dropped (the subsystems removed in the
2026-09-28 cleanup, inert in these runs), the one rename
(`rblapsum_temperature` -> `temperature`), and the fields added since at the
defaults that reproduce the old behaviour (`share_projections: false`,
`rblapsum_surrogate_scope: pool`, `bottleneck_decoder_scale: none`,
`bottleneck_init: standard`, `final_val_batches: 0`, ...).

Run one as it is (self-contained, no `_base_`):

    python scripts/train_guard.py --config configs/cr_kj/ma_cr_rbk_k32_j32_pnorm.yaml

Same seed (1337), data, schedule and architecture as the originals; bf16
matmuls differ across GPU generations, so curves track the originals' rather
than coincide.

## The K=32, T=2 sweep (requested 2026-10-04)

From the four code-residual cells `ma_cr_rbk_k32_j<J>_t2.yaml` (J = 32, 96,
224, 480), one more change each, written by
`scripts/make_cr_kj_configs.py --sweep configs/cr_kj`:

| suffix | change |
| --- | --- |
| `_ss08`, `_ss06`, `_ss04` | `rblapsum_support_scale` 0.8, 0.6, 0.4 (the coefficient of the surrogate support term in the backward) |
| `_fo` | `rblapsum_surrogate_scope: first_order` at support scale 1.0 |

Sixteen runs, compared in the note with the stream-carried T=2 cells at K=32.
