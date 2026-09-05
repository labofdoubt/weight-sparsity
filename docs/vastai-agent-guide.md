# Working on a vast.ai box for this project

Written for another agent picking up an instance from scratch. Covers setup
(repo, data, TensorBoard, Drive sync), then the parts specific to loading
checkpoints and instrumenting the model. Most of this is here because it went
wrong once.

---

## 0. What kind of machine this is

A vast.ai instance is an **unprivileged container** on a shared host, not a
machine. Two consequences bite immediately:

**You do not have the cores `nproc` reports.** `/proc` is not namespaced, so
`nproc`, `sched_getaffinity` and `/proc/loadavg` all show *host-wide* values.
Your real entitlement is the cgroup quota:

```bash
cat /sys/fs/cgroup/cpu.max            # "quota period" -> quota/period cores
# or cgroup v1:
echo $(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / \
         $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
```

Torch reads `nproc` and spawns that many compute threads. On one box that was
192 threads against an 18-core quota: **34% of all scheduling periods were
throttled**, and the test suite went from 18s to over 10 minutes. Always cap:

```bash
export OMP_NUM_THREADS=<quota/workers> MKL_NUM_THREADS=$OMP_NUM_THREADS
```

Capping was measured *faster and 4x cheaper in CPU* than not capping — the
extra threads were pure contention. `scripts/gpu_queue.sh` derives this itself.

**`/workspace` survives stop/start but not destruction.** Nothing else on the
box does. Back up as you go; do not plan to copy things off at the end.

The image also ships an agent guide at `/etc/vast-agents-guide.md` describing
supervisor, Caddy and the port layout. Worth reading once.

---

## 1. Repo and deps

```bash
source /venv/main/bin/activate
cd /workspace && git clone https://github.com/labofdoubt/weight-sparsity.git
cd weight-sparsity
uv pip install -q transformers datasets tokenizers scikit-learn pandas \
                  pyarrow matplotlib tensorboard
uv pip install -q -e . --no-deps      # torch is already in the image
mkdir -p /workspace/{runs,hf_cache,data,plots}
```

`benchmark_data/` (the frozen interpretability dataset) comes with the clone —
it is committed, not downloaded.

**Do not install into `/venv/main` casually if training is running there.** A
dependency resolution can pull a different numpy or torch out from under a live
job. If you only need dev tooling, note the system `python3` often already has
`pytest`, and it is pure Python, so you can borrow it:

```bash
PYTHONPATH=src:/usr/lib/python3/dist-packages python -m pytest tests/ -q
```

Expect **1 failure**: `test_topk.py::test_compiled_model_matches_eager`. It is
pre-existing, it is not yours, and §11 explains what it means.

---

## 2. Google Drive (rclone)

**The user configures this, not you.** `rclone.conf` holds a live OAuth refresh
token; it must never pass through a conversation. Ask them to run, from their
own machine:

```bash
ssh -p <port> root@<host> 'mkdir -p /root/.config/rclone'
scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf
```

Then **verify with a write round-trip**, not just a listing. Read access proves
the token works; it does not prove you can create files:

```bash
rclone lsf gdrive:weight-sparsity/
echo test > /tmp/_w.txt
rclone copy /tmp/_w.txt gdrive:weight-sparsity/_selftest/
rclone cat gdrive:weight-sparsity/_selftest/_w.txt      # must echo back
rclone delete gdrive:weight-sparsity/_selftest/_w.txt
rclone rmdir gdrive:weight-sparsity/_selftest
```

Known quirks:

* The remote currently uses rclone's **shared** `client_id` (the key is present
  in the config but empty). It works, but rclone warns it is being retired
  during 2026. Do not claim it is a private quota — checking that the line
  exists is not the same as checking it has a value.
* **Google Drive allows duplicate directory names.** `runs_taiwan/` has two
  directories with the same name, one holding the checkpoints and one nearly
  empty. A naive per-directory listing can hit the wrong one and report a run
  as missing when it is not. Count recursively (`rclone lsf -R`), and use
  `rclone dedupe --dedupe-mode list <path>` when something looks absent.
* Use `--drive-chunk-size 128M` for large files. Without it, throughput was
  1.4 MB/s against 10 MB/s with it.

---

## 3. Data

**Pull it from Drive. Do not relay it through anyone's laptop.**

```bash
tmux new-session -d -s datadl \
  "rclone copy gdrive:weight-sparsity/data /workspace/data --drive-chunk-size 128M"
```

~2 minutes for 948 MB. A laptop-relayed `ssh ... | ssh ...` transfer of the same
data ran at 350 KB/s and then **died silently when the driving task ended**,
leaving `train.bin` truncated at 797 of 948 MB with `val.bin` missing. Training
would have read the truncated memmap without complaint.

**Always verify:**

```python
import hashlib, json, os
for f, exp in (("train.bin", 947984472), ("val.bin", 9531836)):
    assert os.path.getsize(f"/workspace/data/tinystories/{f}") == exp, f
d = hashlib.sha256(open("/workspace/data/tinystories/val.bin","rb").read(1<<22)).hexdigest()[:16]
w = json.load(open("/workspace/weight-sparsity/benchmark_data/split_story_ids.json"))["val_bin_digest"]
assert d == w, (d, w)      # must be dc382450504b59ae
```

The digest matters beyond corruption: the frozen interpretability benchmark
indexes tokens by position in `val.bin`, so a different stream silently
invalidates every concept label.

---

## 4. TensorBoard

**It is already running.** The image starts it under supervisor on port 16006
with `--logdir /workspace`, fronted by Caddy on 6006 with token auth. Do not
start a second one — you will get "port already in use" and two indexes.

Repoint it at the runs directory so it does not walk the 948 MB corpus and the
HF cache on every reload:

```bash
echo "TENSORBOARD_LOG_DIR=/workspace/runs" >> /etc/environment
supervisorctl restart tensorboard
```

The user tunnels it themselves:

```bash
ssh -p <port> root@<host> -L 16006:localhost:16006     # then http://localhost:16006
```

TensorBoard names a run by its **path relative to `--logdir`**, so runs from two
machines with the same name merge into one interleaved series — silently, and it
looks like a noisy curve rather than an error. Keep other machines' events in a
subdirectory (`/workspace/runs/<machine>/...`). `scripts/tb_mirror.sh` does this
via Drive, since two vast boxes have no SSH trust with each other.

---

## 5. Backups

Two watchers, because the tiers have very different sizes and you do not want
event files waiting behind a 1.4 GB checkpoint upload:

```bash
cd /workspace/weight-sparsity
tmux new-session -d -s backup_tb \
  "bash scripts/backup_watch.sh /workspace/runs gdrive:weight-sparsity/runs_<name> 60"
tmux new-session -d -s backup_ckpt \
  "bash scripts/backup_watch.sh /workspace/runs gdrive:weight-sparsity/runs_<name> 600 --all-checkpoints"
tmux new-session -d -s backup_analysis \
  "bash scripts/backup_analysis_watch.sh /workspace/analysis gdrive:weight-sparsity/analysis_<name> 600"
```

* Use a **distinct remote prefix per machine**; mixing them makes provenance
  unrecoverable.
* **The runs watchers do not cover `/workspace/analysis`.** `backup_runs.sh`'s
  filter list admits logs/configs/events and nothing else, so the probe and
  score datasets -- 15+ GB of `.npy` that take hours of GPU time to recompute --
  need the third watcher (`backup_analysis_watch.sh`, a plain filtered
  `rclone copy`). Restore on a fresh box with
  `rclone copy gdrive:weight-sparsity/analysis_<name> /workspace/analysis
  --drive-chunk-size 128M`; the streamlit viewer reads the restored files
  unchanged (see `analysis/README.md`, "Surviving instance destruction").
* It is `rclone copy`, never `sync`, so local deletions are not mirrored
  upward — that is what makes the checkpoint pruner safe.
* The checkpoint interval must stay well inside the local retention window
  (`keep_last_checkpoints=3` x `checkpoint_every_steps=2000` x ~600 ms ≈ 60 min),
  or intermediate checkpoints are pruned before they are uploaded.

**Disk fills faster than you expect.** Two runs once died inside `torch.save`
with `RuntimeError: unexpected pos` / `basic_ios::clear: iostream error` — that
is a full disk, not a code fault. `scripts/clean_checkpoints.sh <runs> <remote>
[--keep-latest-only]` prunes, and refuses to delete anything it cannot first
find on the remote.

---

## 6. Getting checkpoints back

Drive layout:

| prefix | what |
| --- | --- |
| `runs_server1/`, `runs/` | the earliest machines (10x640 geometry) |
| `runs_server3/` | sweden — 20 residual-placement runs, 8x768 |
| `runs_taiwan/` | 4 init/dense-control runs, 8x768 |
| `runs_dc/` | the `dc_*` 20k-step runs (8x768 soft j-sweep, hard k-sweep) |
| `runs_lens_1/` | current machine — probe / gainsweep dynamics runs |
| `analysis_lens_1/` | the analysis datasets for the streamlit viewer (§8) |
| `benchmark_data/` | also in git |
| `data/` | tokenized corpus |
| `interpretability_results/`, `plots/` | analysis output |

Each run holds `latest.pt` (~1.4 GB), several `ckpt_step*.pt`, `metrics.jsonl`,
`config.json`, `config.yaml`, `summary.json`, `tb/`.

**Pull selectively** — all of them is ~280 GB:

```bash
rclone copy gdrive:weight-sparsity/runs_server3/<run>/latest.pt /workspace/ckpt/<run>/
# metadata only, for picking which ones you want (a few MB total):
rclone copy gdrive:weight-sparsity/runs_server3 /workspace/meta \
  --filter '+ */config.json' --filter '+ */summary.json' \
  --filter '+ */metrics.jsonl' --filter '- *'
```

A run without `summary.json` did not finish — check before treating it as a
result.

---

## 7. Loading a checkpoint and instrumenting it

```python
from wsparse.train import load_for_inference
model, cfg, _ = load_for_inference("/workspace/ckpt/<run>/latest.pt", device="cuda")
model.eval()
```

It rebuilds the model from the config stored *inside* the checkpoint, so the
architecture always matches the weights. `cfg.activation_bottleneck` tells you
the geometry actually used — trust it over the run name.

To reach the bottleneck internals, hook the module. `x` is its input, `z` the
sparse code that reaches the decoder, `x_hat` its output:

```python
from wsparse.bottleneck.controller import _PLACEMENT_ATTR
block = model.blocks[layer]
mod   = getattr(block, _PLACEMENT_ATTR[cfg.activation_bottleneck.placement])

grab = {}
mod.register_forward_pre_hook(lambda m, i: grab.__setitem__("x", i[0].detach()))
mod.register_forward_hook(lambda m, i, o: grab.__setitem__("x_hat", o.detach()))
mod.gate.register_forward_hook(lambda m, i, o: grab.__setitem__("z", o.detach()))
```

The decoder is `mod.out_proj.weight` `(d_model, n_features)`, columns are feature
directions; `mod.output_scale` is a frozen gain when `calibrate_output` was on.
`scripts/../interpretability/extract_bottleneck_activations.py` is a worked
example, including how it batches stories and pins target token positions.

**For gradients**, note the gate is a custom autograd Function with a
hand-written backward. Things that will surprise you:

* `surrogate_active()` returns **False** in eval or under `no_grad`, so the
  surrogate path is skipped entirely — measure gradients in `train()` mode with
  grad enabled, or you are measuring the hard mask.
* Under `surrogate_mode=hard` the J candidates receive **exactly** zero; the
  logged `grad_inactive` is a literal `0.0`. Under a lapsum mode it is small but
  nonzero (~1e-8 late in training). On a linear TensorBoard axis that renders as
  a flat line at zero — use a log axis before concluding it is off.
* `j` is **inert** under `surrogate_mode=hard`: verified bit-identical output and
  gradients for j = 1, 64, 256, 1504. It only changes the diagnostics window.
* `n_eff` / `boundary_mode` / `one_sided_weight_mode` are **inert** under
  `lapsum_scheduled` — there the temperature comes from the schedule, and
  `n_eff_realized` drifting far from the configured `n_eff` is expected.

---

## 8. The analysis pipeline (probes -> streamlit -> Drive)

`analysis/` is the project's second half: four scripts that produce datasets
under `/workspace/analysis/{scores,probe,gain,wnorm}`, two inspection CLIs, and
one streamlit viewer over them. `analysis/README.md` is the complete manual; the shape of it:

* **Two ways to get data.** `extract_bottleneck_scores.py` walks *saved*
  checkpoints (every 2000 steps). The three probes (`probe_early_training.py`,
  `probe_gain_sweep.py`, `probe_weight_norms.py`) re-run training from a run's
  own `config.json` with the same seed and measure every 10 steps through
  `train()`'s `on_step` hook, saving no checkpoints -- steps 0..1000 exist
  nowhere else. `--set a.b=c` overrides let them probe configs that never
  existed as runs.
* **The two probe invariants.** A probe must not perturb the run (the hook fires
  before that step's `zero_grad`, so probe backwards are provably discarded, and
  gate buffers / RNG are restored), and it must not touch `max_steps` --
  schedules are defined over it (§9); stop early by raising `StopProbing`
  instead. Gradient probes run in `train()` mode on purpose: `surrogate_active()`
  is False in eval / `no_grad` (§7), so an eval-mode probe measures the hard
  mask rather than the surrogate.
* **The viewer** runs on the box as the `score_explorer` supervisor service on
  `127.0.0.1:8501`; tunnel it like TensorBoard (`-L 8501:localhost:8501`). It
  discovers datasets by listing its three directories with a 30 s cache, so a
  new run appears ~30 s after its file lands -- no restart, no app change.
  Adding a new *kind* of data means a new page: README §7 has the pattern.
  Caveats baked into it deliberately: `j` is forced to 0 for hard-gate runs
  (inert in training -- §7), and the LapSum barrier band is reconstructed
  offline because `scheduled_temperature` is a non-persistent buffer absent
  from checkpoints.
* **`gain/` JSONs are not shown in the app** -- they feed TensorBoard
  (`lens_1/gainsweep_*`, via `--tb-dir`) and ad-hoc comparison scripts.
* **Drive sync**: the third backup watcher (§5) mirrors `/workspace/analysis`
  -> `gdrive:weight-sparsity/analysis_<machine>` every 10 min. The runs
  watchers do **not** cover it, and the datasets are 15+ GB that take hours of
  GPU time to recompute. Standing the viewer up on a fresh box without
  recomputing anything is one `rclone copy` plus two `cp` for the supervisor
  service files kept in `scripts/` -- README §8 ("Surviving instance
  destruction") has the exact commands. The viewer needs no corpus, no
  checkpoints and no GPU: everything it reads travels inside the datasets and
  their meta JSONs.

---

## 9. Config traps worth knowing

* **`placement`** — `pre_mlp` / `post_attn` / `post_mlp` sit inside a residual
  branch (the skip routes around them). `residual` / `residual_out` replace the
  stream at the head / tail of a block, with nothing routing around, so the
  bottleneck's per-layer gain compounds as `g^n_layers`. With `layers=all`,
  `residual` and `residual_out` differ in only one insertion point out of
  `n_layers + 1`.
* **The bottleneck does not inherit the model's conventions.** It is spliced in
  *after* `_init_weights` runs, so `init_scheme` / `init_std` / `init_gain` never
  touch it, and `activation_bottleneck.bias` is a separate field that defaults
  to `True` while `model.bias` defaults to `False`.
* **`init_mode`** controls the bottleneck's own init. `default` (PyTorch) scales
  the decoder for fan-in `n_features` though TopK delivers only `k` non-zeros;
  measured `std(out)/std(in)` = 0.128 at k=32, N=1536 against 1.006 for
  `sqrt_k_selection_corrected`. On a stream placement that 8x attenuation
  compounds and the run collapses — reproduced deliberately: 5.98 with 93.8%
  dead features, against 1.83 for the same config with a corrected init.
* **`init_std_embedding` does not default to `init_std`** — it defaults to
  `1/sqrt(2)`.
* **`init_std_unembedding` is rejected when `tie_embeddings=true`** (one matrix).
* **`logit_scale=auto`** matters only for a tied or explicitly scaled head; at
  the default unembedding std it is a no-op.
* **`pos_encoding`** — `"learned"` (default: the absolute position table) or
  `"rope"` (rotary, applied to q/k in every attention layer). Checkpoints saved
  before the field existed resolve to `learned` and load bit-identically
  (verified: CE diff 0.00e+00 on `dc_rout_soft_k32_j64@20000`). Under rope there
  is **no `pos_emb` at all** — the parameter count drops by
  `max_seq_len * d_model` and `init_std_pos` is unused, so the
  "tok+pos has unit variance" reasoning behind `DEFAULT_STD_EMBEDDING` does not
  apply. Measured cost on the 4090 at 24x512 bf16: +5% forward, +3-4%
  forward+backward. The rope cos/sin cache is a non-persistent buffer, so
  state_dicts stay interchangeable within a mode.

### `decouple` — magnitude-direction decoupling (arXiv:2606.25971)

`model.decouple=true` trains every matrix as a fixed-Frobenius-norm direction
times learnable per-row/per-column softplus gains (`model.decouple_gains`:
`row_col` default, `up_down` for the one-sided nGPT-style split), with
embeddings and the LM head held at unit L2 row norm and a fixed `sqrt(d)`
embedding upscale in the forward. Implemented in `src/wsparse/decouple.py` as an
*optimizer-side* method: the model stores ordinary fused weights, so
checkpoints stay plain and the forward/backward is untouched.

Things it silently overrides, by design: **every** init field (`init_scheme`,
`init_std*`, `init_gain`, `init_scale_residual`, and the bottleneck's
`init_mode` family — the init is entrywise `1/sqrt(d_model)` projected onto
`c_F = sqrt(d_out*d_in/d_model)`), and **weight decay** — a configured
`train.weight_decay` is *ignored*, not rewritten: no parameter group receives
decay and the update kernel has no decay term, but the dumped `config.yaml`
still shows the configured number, so do not read decay settings off a
decoupled run's config. It requires `pos_encoding="rope"` and
`logit_scale="none"`, and refuses `sparsity.enabled=true` and float16.

Measured overhead on the 4090 at 24x512 bf16 (all of it in `optimizer.step`,
which is the point of the fused design): full training step **+2.0%** plain LM,
**+6.1%** with the bottleneck's extra 18.9M matrix params; the optimizer step
alone is 1.25x / 1.9x. In real training the observed cost was smaller still
(+0.4% plain, +1.5% with bottleneck at grad-accum 4): accumulation amortizes
the optimizer step.

Implementation facts that will otherwise confuse you (`src/wsparse/decouple.py`):

* **The gains are optimizer state, not model parameters.** `model.parameters()`
  and the logged parameter count are identical to a non-decoupled run, the
  checkpoint's `"model"` blob holds plain fused weights, and `load_for_inference`
  needs nothing special. Per matrix the optimizer state carries `raw_grow` /
  `raw_gcol` (softplus-raw, initialized to `ln(e-1)` so every gain starts at
  exactly 1), their Adam moments, and the sphere radius `c_f`. The kernel is
  hand-rolled: moments are named `m`/`v`, not stock AdamW's
  `exp_avg`/`exp_avg_sq` -- grepping the state for the stock names finds nothing.
* **The fused `W` is not on the c_F sphere -- only the recovered direction is.**
  `W = diag(softplus(raw_grow)) @ W_hat @ diag(softplus(raw_gcol))`, so once the
  gains move, `W.norm()` drifting off `c_F` is the method working, not the
  constraint breaking. Divide the gains out before checking (recipe below).
  Embedding and LM-head rows *are* exactly unit-norm in the fused checkpoint
  (they carry no gains) -- the quickest fingerprint of a decoupled checkpoint
  after `config.json`'s `model.decouple`.
* **`c_f` is captured on first sight of each parameter, not recomputed from its
  shape.** The optimizer trusts that `md_init_` already ran (gains are exactly 1
  then, so `||W|| == ||W_hat||`). Consequence: "resuming" by loading only the
  model weights into a fresh optimizer silently redefines the constraint --
  gains reset to 1 and `c_f` is captured from the *gained* fused norm -- with no
  error anywhere. A real resume must restore `payload["optimizer"]`, which the
  normal `train.resume` path does (the roundtrip is tested). Relatedly: on a
  resumed run the startup line about re-initializing matrices still prints
  (`md_init_` runs before the checkpoint load); the loaded weights and optimizer
  state overwrite all of it, so it is cosmetic.
* **Gains follow the matrix LR.** `set_lr` writes every group each step; there
  is no separate gain-LR knob, and warmup applies to gains too. (The paper runs
  warmup-free with its own embedding LR -- there that is configuration, not
  code: `train.warmup_steps=0`.)
* **`up_down` picks the gain side from the shape:** `d_out >= d_in` -> row gain,
  else column gain; square matrices (`attn.proj`) count as up/row. `row_col`
  (the default) gives every matrix both. The bottleneck's `in_proj`/`out_proj`
  are ordinary md matrices and get gains like everything else.
* Fused-weight analysis stays valid but reads differently: in
  `analysis/probe_weight_norms.py` output, a decoupled run's Frobenius growth is
  carried entirely by the gains (the direction norm is pinned at `c_f`).

To inspect a decoupled checkpoint's gains and directions, do not parse the raw
`payload["optimizer"]` dict (its state is keyed by integer position within
groups); rebuild and let the optimizer map it:

```python
import torch, torch.nn.functional as F
from wsparse.train import load_for_inference
from wsparse.decouple import build_decoupled_optimizer

model, cfg, _ = load_for_inference(path, device="cpu")
payload = torch.load(path, map_location="cpu", weights_only=False)
opt = build_decoupled_optimizer(model, cfg.train, gain_mode=cfg.model.decouple_gains)
opt.load_state_dict(payload["optimizer"])

W  = model.blocks[0].mlp.fc1.weight
st = opt.state[W]                       # raw_grow, raw_gcol, c_f, moments
w_hat = W.detach().clone()
if "raw_grow" in st:                    # under up_down one of the two is absent
    w_hat /= F.softplus(st["raw_grow"]).unsqueeze(1)
if "raw_gcol" in st:
    w_hat /= F.softplus(st["raw_gcol"]).unsqueeze(0)
assert (w_hat.norm() - st["c_f"]).abs() < 1e-3 * st["c_f"]
```

Worked examples: `lens_1/gainsweep_rope_LMropeMD` and `..._BNropeMD` (1000-step
dynamics against the `..._LMrope` / `..._BNrope` baselines, identical configs
apart from the flag; probe JSONs in `/workspace/analysis/gain/`).

### Every schedule is defined over `train.max_steps`

Both the learning rate (`optim.lr_at`) and the bottleneck temperature
(`build_schedule`, whose `anneal_steps` defaults to `max_steps - warmup_steps`)
are parameterised by `max_steps`, not by wall-clock steps. **So shortening
`max_steps` to stop a run early does not truncate the schedules -- it compresses
them**, and the run you get is not the first N steps of the run you wanted.

If you need to stop early (a probe run, a quick repro), leave `max_steps` alone
and stop by other means. `analysis/probe_early_training.py` raises a
`StopProbing` exception from `train()`'s `on_step` hook for exactly this reason.
Sanity check: with `lr=6e-4` and `warmup_steps=500`, step 1 must log
`lr 1.20e-06`. If it logs something larger, the schedule has been rescaled.

### What the `dc_*` runs actually use

All of `dc_rout_soft_k32_{j32,j64,j128}`, `dc_rout_hard_k32_selcorr` and
`res_hard_selcorr_k64` share these:

| field | value |
| --- | --- |
| `lr` / `lr_schedule` | `6e-4` / `cosine` |
| `warmup_steps` / `min_lr_ratio` | 500 / 0.1 (so it decays to `6e-5`) |
| `max_steps` | 20000 |
| `betas` / `weight_decay` / `grad_clip` | (0.9, 0.95) / **0.1** / 1.0 |
| `batch_size` / `micro_batch_size` | 96 / 24 (so accum 4) |
| `seq_len` / `dtype` / `compile` | 512 / `bfloat16` / `False` |

Consequence worth knowing before reading any early-training result: with warmup
at 500 of 20000, the first ~1000 steps sit at or barely below **peak** LR (6e-4
at step 500, still 5.99e-4 at step 1000 -- the cosine has decayed 0.2%). Nothing
in that window can be attributed to LR decay.

### Weight decay reaches the bottleneck

`build_optimizer` groups parameters **purely by `p.dim() >= 2`** -- there are no
name-based exemptions. So the bottleneck's `in_proj` / `out_proj` are decayed at
`weight_decay` exactly like `attn.qkv` and `mlp.fc1/fc2`; only biases and RMSNorm
scales escape (13,056 parameters out of 114.5M). Do not assume the bottleneck is
exempt -- and read any weight-norm growth as happening *against* a 0.1 decay.

### The soft runs' temperature never anneals

`dc_rout_soft_*` set `temperature_schedule="constant"` with
`temperature_start=1.0`, so `Schedule.__call__` returns `start` and **ignores
`temperature_end`** entirely. The configs also carry `temperature_end=0.02`,
which reads like an intent to anneal 1.0 -> 0.02; it never happens.
`temperature_anneal_steps`, `temperature_power` and `fixed_temperature` are
inert there too.

With `temperature_scale_mode="relative"` the effective temperature is therefore
`t = 1.0 * std(top-(k+j) scores)` -- no time dependence and no absolute
reference, so the surrogate's operating point is pinned to whatever the score
scale happens to be. `t/scale` is measurably constant across the whole
checkpoint ladder. That matters: see `docs/activation-amplification.md`, where
forcing `temperature_scale_mode=absolute` cuts the activation tail 14.7x.

The hard runs declare `exponential` 0.5 -> 0.02, which is equally inert -- a hard
gate never solves a barrier and never reads the temperature at all.

---

## 10. Operational habits that were learned the hard way

* **Run anything long in `tmux` on the server.** Backgrounded commands driven
  from an agent session die when the task is reaped, mid-transfer, without an
  error. This destroyed one dataset transfer and nearly a training chain.
* **Chain work with `tmux has-session -t=NAME`.** The `-t=` form is exact match;
  a bare `-t NAME` prefix-matches and will wait on `NAME2`.
* **Preflight before a queued job runs, not after.** Check the commit actually
  has the feature you rely on and abort loudly otherwise. A queued chain once
  ran four jobs that all died instantly with `python: command not found`. It
  happened again with `python: can't open file` -- the scripts were reorganised
  into a new directory while a job sat queued against the old path. Anything
  queued behind a `tmux has-session` wait is running against the filesystem as
  it will be *later*, not as it is when you write the command.
* **Do not filter a queued job's output down to what you expect.** The second
  instance above was invisible for a full queue cycle because the runner piped
  through `grep -E "Traceback|Error"`, and `python: can't open file ...` matches
  neither. Log the tail of everything, or grep for success and treat silence as
  failure.
* **Verify sizes and digests after any transfer.** "It looked slow" and "it
  failed" are indistinguishable without a check.
* `torch.multinomial` raises a **device-side assert** on non-finite logits, so a
  collapsed model crashes in the *sampling* callback rather than in training.
  Read that as a symptom, not the cause.

---

## 11. The one test that always fails

Every fresh run of the suite ends with

```
FAILED tests/test_topk.py::test_compiled_model_matches_eager - assert False
1 failed, 433 passed
```

**You did not break it.** It reproduces on a pristine `git clone` of the repo at
`c04a171` with nothing modified, on `torch 2.11.0+cu128`. Do not go looking for
your own change; do not "fix" it by loosening the tolerance.

**But it is not a rounding artifact, so do not dismiss it either.** The test
compares an eager backward against a `torch.compile`d one. Measured on this box:

| | loss | `weight.grad` norm | `s.grad` norm |
| --- | --- | --- | --- |
| eager | 1.7309851646 | 1.1e-02 .. 2.1e-02 | ~5.3e-05 |
| compiled | 1.7309851646 | **0.0** | **0.0** |

The forward is fine — the losses agree to every printed digit, and the selected
support is bit-identical (1229 of 4096 weights in each layer). The *backward*
produces **exactly zero** gradient for every sparsity-wrapped parameter. So
under `torch.compile` the weight-sparsity layers would train nothing at all,
silently. The assertion is doing its job; the tolerance is not the problem.

The likely cause is the very thing the test's docstring describes:
`supports()` is `torch.compiler.disable`d so that the version-counter cache does
not force a recompile every optimizer step. On torch 2.11 that graph break
appears to leave the hand-written backward disconnected from the autograd graph
rather than merely un-fused.

**Why it is nonetheless safe to carry on:**

* `train.compile` defaults to `False` (`config.py`), and **every run in this
  project so far was trained with `compile=False`** — checked across
  `dc_rout_soft_k32_{j32,j64,j128}`, `dc_rout_hard_k32_selcorr` and
  `res_hard_selcorr_k64`. No existing result is affected.
* The activation-bottleneck runs have `sparsity.enabled = False` anyway, so the
  wrapped layers this test exercises are not even installed in them.

**What this does mean:** do not set `train.compile=true` while
`sparsity.enabled=true` on this torch version. If you ever want compile, verify
first that `layer.weight.grad` is non-zero after one step — the failure mode is
silent, and a run would simply not learn its masks while still reporting a
falling loss.
