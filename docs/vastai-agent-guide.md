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
```

* Use a **distinct remote prefix per machine**; mixing them makes provenance
  unrecoverable.
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
| `runs_dc/` | current machine |
| `benchmark_data/` | also in git |
| `data/` | tokenised corpus |
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

## 8. Config traps worth knowing

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

---

## 9. Operational habits that were learned the hard way

* **Run anything long in `tmux` on the server.** Backgrounded commands driven
  from an agent session die when the task is reaped, mid-transfer, without an
  error. This destroyed one dataset transfer and nearly a training chain.
* **Chain work with `tmux has-session -t=NAME`.** The `-t=` form is exact match;
  a bare `-t NAME` prefix-matches and will wait on `NAME2`.
* **Preflight before a queued job runs, not after.** Check the commit actually
  has the feature you rely on and abort loudly otherwise. A queued chain once
  ran four jobs that all died instantly with `python: command not found`.
* **Verify sizes and digests after any transfer.** "It looked slow" and "it
  failed" are indistinguishable without a check.
* **A run stuck at a plateau with a high `feature_dead_frac` is collapsed**, not
  slow. Kill it. `bottleneck/feature_dead_frac` and `feature_usage_entropy` are
  the early indicators; loss and `budget_residual` look healthy throughout.
* `torch.multinomial` raises a **device-side assert** on non-finite logits, so a
  collapsed model crashes in the *sampling* callback rather than in training.
  Read that as a symptom, not the cause.
