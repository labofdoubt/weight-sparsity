# Bottleneck analysis pipeline

Scripts and one streamlit viewer for looking at what a stream-placed activation
bottleneck actually does: the scores its gate ranks on, the gradients that reach
them, and the weight norms of the blocks around it.

Separate from [`../interpretability/`](../interpretability/), which is the frozen
concept benchmark.

Everything writes to `/workspace/analysis/<subdir>` on the box and is read back
memory-mapped, so the viewer only touches the cell it draws.

| script | what it produces |
| --- | --- |
| `extract_bottleneck_scores.py` | scores from saved checkpoints -> `scores/<run>.npy` |
| `probe_early_training.py` | scores **and both gradients** during a fresh run, no checkpoints kept -> `probe/<run>.{score,g_ztilde,g_z}.npy` |
| `probe_gain_sweep.py` | per-layer gain / `enc` / `keep` / `cos` for a k or j sweep -> `gain/<name>.json` |
| `probe_weight_norms.py` | every block's weight-matrix norms every N steps -> `wnorm/<name>.json` |
| `inspect_bottleneck_scores.py` | prints the distribution facts that decide how to plot |
| `score_explorer.py` | the streamlit viewer over all of the above |

# Bottleneck score explorer

A second, separate pipeline from the concept benchmark above: it looks at the
gate's *ranking scores* and its gradients rather than at concepts. Three scripts
feed one streamlit viewer.

## 1. Scores from saved checkpoints

```bash
python interpretability/extract_bottleneck_scores.py \
    --ckpt-dir /workspace/ckpt/<run> \
    --data-dir /workspace/data/tinystories \
    --out-dir /workspace/analysis/scores \
    --batch 8 --positions 128
```

Walks every `ckpt_step*.pt` in the directory and stores the pre-TopK score the
gate ranks on, captured as a forward **pre**-hook on `mod.gate` — its first
positional argument is the ranking signal in *every* `selection_mode`, so the
hook is placement- and mode-agnostic. Writes
`<run>.npy` `(ckpt, bottleneck, sequence, position, n_features)` float32 plus a
`<run>.json` of steps, labels, k, j, token ids and the temperature schedule.

One fixed batch (`deterministic_offset`) is reused for every checkpoint and
every run, so a difference between two cells is a difference in the model and
never in the data.

## 2. Early training, without keeping checkpoints

`checkpoint_every_steps` is 2000, so steps 0..1000 have no saved weights, and
keeping 100 checkpoints would cost ~140 GB per run. Instead:

```bash
python interpretability/probe_early_training.py \
    --config /workspace/ckpt/<run>/config.json \
    --data-dir /workspace/data/tinystories \
    --out-dir /workspace/analysis/probe \
    --tb-dir /workspace/runs/<machine> \
    --steps 1000 --probe-every 10 --batch 4 --positions 64
```

Re-runs training from the run's own `config.json` with the same seed, and
measures the live model on a fixed held-out batch every 10 steps (including
step 0, at initialization). Records three arrays: `score`, `g_ztilde`
(`dL/d~z`, at the gate's output, dense) and `g_z` (`dL/dz`, at its input,
after the LapSum surrogate, exactly zero outside Top(K+J)).

Two things it is careful about, both of which would silently invalidate the
result:

* **The probe must not train the model.** It runs its own backward, so it is
  called from `train`'s `on_step` hook *before* that step's
  `optimizer.zero_grad(set_to_none=True)` — which is then what guarantees the
  probe's gradients cannot reach the optimizer. It also restores the RNG state
  and every buffer the gate mutates while measuring (`usage_ema`,
  `usage_steps`, the diagnostics dicts). Verified: a probed run's step-20 CE
  falls inside the spread of two *identical* unprobed runs (11.0932 / 11.0938 /
  11.0943), so the perturbation is below the training loop's own
  nondeterminism. It is **not** bit-identical — TF32 and non-deterministic
  backward kernels mean no two runs are.
* **The schedules must not be rescaled.** `lr_at` decays cosine over
  `train.max_steps` and the temperature anneals over `max_steps - warmup`, so
  lowering `max_steps` to stop early would compress the whole decay into the
  probed window and reproduce nothing. `max_steps` is left alone and the run
  stops by raising `StopProbing` from the hook.

It also runs in `train()` mode on purpose: `surrogate_active()` is False in eval
and under `no_grad`, so a probe in eval mode would measure the *hard* mask's
gradient rather than the surrogate's.

## 3. Characterize before plotting

```bash
python interpretability/inspect_bottleneck_scores.py \
    --scores /workspace/analysis/scores/<run>.npy
```

Prints the dynamic range per band, how visible the TopK/J boundary is, how much
of a linear axis each band occupies, TopK identity persistence, and the support
size. This is what decided the viewer's format: the selection margin is
0.002–0.008 of the TopK span, i.e. sub-pixel, so a bare number line cannot show
the boundary at all.

## 4. The viewer

```bash
SCORES_DIR=/workspace/analysis/scores PROBE_DIR=/workspace/analysis/probe \
  streamlit run interpretability/score_explorer.py
```

On the vast.ai box it runs as a supervisor service on `127.0.0.1:8501`
(`/opt/supervisor-scripts/score_explorer.sh`), reached by SSH local forwarding:

```bash
ssh -p <port> root@<host> -L 8501:localhost:8501
```

`j` is forced to 0 for `surrogate_mode=hard` runs: their config still carries a
`j`, but it is inert (bit-identical for j = 1..1504), so those ranks get exactly
zero gradient like `rest` and painting them as a live band would be wrong.

The LapSum barrier `b` and the `b ± t` window are reconstructed offline, because
the gate's `scheduled_temperature` buffer is `persistent=False` and so is absent
from checkpoints: `t = schedule(step) * std(top-(k+j) scores)` for
`scale_mode="relative"`, then `b` solves `sum F((r-b)/t) = k` — budget `k`, not
`k+j` — via the project's own `lapsum_barrier_sorted`. Validated against the
values training logged: 0.7% on `b` and 2.2% on `t` for `..._j64`, with the
budget identity holding to 1e-5.
