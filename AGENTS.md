# AGENTS.md

Instructions for coding agents in this repo. Loaded into every session, so it
holds only what applies broadly and stays under ~200 lines; reasons and details
are in `docs/vastai-agent-guide.md` ("§N" refers to its sections).

## How the work is organised

- Code, figures and notes are edited in this checkout; training and analysis
  run on rented vast.ai GPU boxes, driven over SSH.
- **Boxes are temporary.** The user rents one when there is work, destroys it
  when there is none, and sets up the next one the same way. There is no
  permanent box and no permanent address: for each box the user gives
  `ssh -p <port> root@<host>` and a short name (pick one if they don't). An
  address from an earlier session or from memory stops connecting once its box
  is gone, so ask for the current one.
- **Google Drive (`gdrive:weight-sparsity/`, via rclone) is the only durable
  storage.** `/workspace` survives a stop/start, nothing survives destruction,
  so back up while working. Each box writes its own `runs_<name>/` and
  `analysis_<name>/` (`rclone lsf gdrive:weight-sparsity/` lists every box's).
- Renting, stopping and destroying boxes is the user's job.

## Setting up a box

Substitute `<host>`, `<port>`, `<name>`. Anything long-running goes in tmux.

1. **CPU threads.** `nproc` shows the whole host; the real quota is in
   `/sys/fs/cgroup/cpu.max` (quota/period cores). For manual launches set
   `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to quota ÷ concurrent jobs (`gpu_queue.sh`
   does it itself); uncapped, the tests once ran over 30× slower (§0).
2. **Repo and deps** (§1). The repo is public; torch comes with the image.
   ```bash
   source /venv/main/bin/activate
   cd /workspace && git clone https://github.com/labofdoubt/weight-sparsity.git && cd weight-sparsity
   uv pip install -q transformers datasets tokenizers scikit-learn pandas \
                     pyarrow matplotlib tensorboard streamlit plotly pytest
   uv pip install -q -e . --no-deps
   mkdir -p /workspace/{runs,analysis,data,hf_cache,plots}
   ```
   If `uv` fails with `invalid peer certificate: UnknownIssuer`, add `--system-certs`.
3. **rclone config** (§2) holds a live Drive token and the client secret. Copy
   it from the local machine, but never open or print it: no `cat`/`head`, no
   `rclone config show`/`dump`, no quoting it in a message. Creating or
   reconnecting the token is a Google login, so the user does that.
   ```bash
   ssh -p <port> root@<host> 'mkdir -p /root/.config/rclone'
   scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf
   ```
   On the box, check for a private client_id without printing it, then prove
   write access (a listing only proves read access):
   ```bash
   grep -qE '^client_id = .+' /root/.config/rclone/rclone.conf && echo "private client_id set"
   echo test > /tmp/_w.txt && rclone copy /tmp/_w.txt gdrive:weight-sparsity/_selftest/
   rclone cat gdrive:weight-sparsity/_selftest/_w.txt    # must print "test"
   rclone delete gdrive:weight-sparsity/_selftest/_w.txt && rclone rmdir gdrive:weight-sparsity/_selftest
   ```
4. **Drive folders**, once, before any watcher starts:
   `rclone mkdir gdrive:weight-sparsity/runs_<name>`, and the same for
   `analysis_<name>`. Watchers that start together against a missing folder
   each create it, and Drive keeps the duplicates (§2).
5. **Data**, straight from Drive, never relayed through a laptop (§3):
   ```bash
   rclone copy gdrive:weight-sparsity/data/tinystories /workspace/data/tinystories --drive-chunk-size 128M
   stat -c %s /workspace/data/tinystories/{train,val}.bin      # 947984472, 9531836
   python -c "import hashlib; print(hashlib.sha256(open('/workspace/data/tinystories/val.bin','rb').read(1<<22)).hexdigest()[:16])"  # dc382450504b59ae
   ```
   The digest also protects the frozen interpretability benchmark, which
   indexes tokens by position in `val.bin`. Pull FineWeb-Edu
   (`data/fineweb_edu_10bt`, ~20 GB) only for FineWeb configs; each `.bin` must
   be exactly 2 bytes × the token count in its `meta.json`.
6. **TensorBoard** already runs under supervisor on box port 16006 (Caddy with
   a token on 6006; 8080 is Jupyter). Don't start another; point it at the runs:
   `echo "TENSORBOARD_LOG_DIR=/workspace/runs" >> /etc/environment && supervisorctl restart tensorboard`.
   The user tunnels with `ssh -p <port> root@<host> -L 16006:localhost:16006`.
   Runs are named by their path under the logdir, so same-named runs from two
   boxes merge silently: other boxes' runs go under `/workspace/runs/<box>/`
   (`scripts/tb_mirror.sh`, §4). A local copy that outlives the box:
   `TB_HOST=<host> TB_SSH_PORT=<port> TB_LOCAL=~/tb-logs/weight-sparsity/<name> bash scripts/tb_local.sh`.
7. **Backup watchers** (§5), one tmux session each, from the repo directory:
   ```bash
   R=gdrive:weight-sparsity/runs_<name>; A=gdrive:weight-sparsity/analysis_<name>
   tmux new -d -s backup_light    "bash scripts/backup_watch.sh /workspace/runs $R 60"
   tmux new -d -s backup_ckpt     "bash scripts/backup_watch.sh /workspace/runs $R 600 --checkpoints-only --all-checkpoints"
   tmux new -d -s backup_analysis "bash scripts/backup_analysis_watch.sh /workspace/analysis $A 600"
   tmux new -d -s disk_guard      "bash scripts/disk_guard_watch.sh /workspace/runs $R 300 40 15"
   ```
   - One writer per file: `backup_light` owns logs, metrics, configs and event
     files, including ones still being written (§5); `backup_ckpt` owns the
     checkpoints. Two uploaders racing on a new file leave duplicates on Drive.
   - The runs watchers skip `/workspace/analysis`, whose `.npy` datasets cost
     hours of GPU time to recompute; hence `backup_analysis`.
   - `disk_guard` takes the interval, then soft and hard free-space thresholds
     in GB (`60 20` on ~1 TB disks). A full disk shows up as `unexpected pos`
     inside `torch.save`.
   - With `--all-checkpoints` the interval must stay well below
     `keep_last_checkpoints × checkpoint_every_steps × step time`, or
     checkpoints are pruned before they are uploaded.
   - ~500M runs (~7 GB checkpoints): run `backup_ckpt` as `7200 --checkpoints-only`
     (latest only, every 2 h), plus one more `--checkpoints-only` push at each
     run's end (§9c).
   - On a slow uplink put `RCLONE_BWLIMIT=3M` in front of the checkpoint
     watcher. If `rclone -v` shows `network is unreachable` for IPv6 addresses,
     put `RCLONE_BIND=0.0.0.0` in front of every rclone command and watcher (§5).
   - Always `rclone copy`, never `sync`.
8. **Analysis viewer**, only if this box serves it: restore `analysis_<name>`
   and install the streamlit service per `analysis/README.md` §10; tunnel
   `-L 8501:localhost:8501`.
9. **Before launching work:** the test suite green (see below), then a short
   smoke run of a shipped config on one GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/train_guard.py --config configs/bn_hard.yaml --stop-step 30 \
       --data.data_dir=/workspace/data/tinystories --train.out_dir=/workspace/smoke --train.run_name=smoke
   ```

## Restarting, rebooting, tearing down

- **Reusing a stopped box:** after `git pull`, delete stale bytecode, or modules
  removed upstream still import from `__pycache__`:
  `find . -name __pycache__ -type d -prune -exec rm -rf {} + && find src tests -type d -empty -delete`.
  Then rerun the tests.
- **After a reboot** the tmux sessions (watchers, queues) are gone, while the
  supervisor TensorBoard comes back. Check `tmux ls`, then relaunch step 7.
- **Before the user destroys a box:** once nothing is writing, let
  `backup_analysis` finish a cycle, stop the watchers, run
  `bash scripts/backup_runs.sh /workspace/runs $R --all-checkpoints` (`R`, `A`
  as in step 7; `--with-checkpoints` under the latest-only policy), then
  `rclone check --one-way` both directories against Drive. Anything missing must be local-only on purpose
  (e.g. `ckpt_step*.pt` under the latest-only policy, mirrors of other boxes)
  or be uploaded first; ask when unsure. Only then report the box safe to destroy.

## Rules that protect results (each has already cost runs)

- **Never lower `train.max_steps` to stop early.** The LR schedule is defined
  over it, so it compresses instead of truncating. Stop with
  `train_guard.py --stop-step`, `train(should_stop=...)`, or `StopProbing` in
  probes. Check: at lr 6e-4 with 500 warmup steps, step 1 logs `lr 1.20e-06` (§9).
- **Measure gradients in `train()` mode with grad enabled.** `surrogate_active()`
  is False in eval and under `no_grad`, so you would measure the hard mask (§7).
  A probe must not perturb the run it observes (§8).
- **Read a run's own `config.json`, not today's defaults.** The
  `rblapsum_boundary_floor` default went 0.1 → 0.0 on 2026-09-21 and the
  `activation_bottleneck.bias` default became False in the 2026-09-28 cleanup,
  while `analysis/kappa_stability.py` still hard-codes `B0 = 0.1` (§9, §9b).
  `load_for_inference` rebuilds the model from the config inside the
  checkpoint; trust `cfg.activation_bottleneck` over the run name (§7).
- **`model.decouple=true` ignores `train.weight_decay`,** although `config.yaml`
  still shows it. Its gains live in optimizer state, so a resume must restore
  `payload["optimizer"]`. Without decouple, weight decay does reach the
  bottleneck's `in_proj`/`out_proj` (§9).
- **The bottleneck does not inherit the model's init.** `init_mode=default`
  attenuates the output ~8× and collapses stream placements (§9).
- **Deep stream bottlenecks** (beyond ~8 layers) need `code_residual` or
  `value_shift` with `post_norm=false`; no scale, init or post-norm fixes them (§9).
  Resume with the run's own `config.yaml`: those options are not in the state_dict.
- **`rblapsum_sf` logs two losses:** `val/ce` is the hard Top-K forward,
  `val_soft/ce` the soft forward it trains (§9).
- **rblapsum stability:** read Pi per block together with `n_eff`, never
  averaged over blocks (§9b).
- **Keep `train.compile=False`.** A graph break around the hand-written gate
  backward can zero gradients silently while the loss keeps falling (§11).
- **A run without `summary.json` did not finish.**
- **Archived configs** mostly reload through migration (§9d); the
  relative-temperature `dc_rout_soft_*` family raises by design. Rebuild those
  from the pre-cleanup tree `e9eada3`.
- **Before landing anything that could move numerics,** run
  `tools/repro_check.py` against the previous commit (§9d) and a real smoke run
  of a shipped config: the oracle cannot see what its own config pins.
- **DDP:** batch sizes are per rank, rank 0 writes every output, and stops go
  through the `should_stop` flag, never an exception on one rank (§9c).

## Working habits (§10)

- Commands backgrounded from an agent session die silently when the session
  ends, so long work runs in tmux on the box. Chain on
  `tmux has-session -t=NAME`: `-t=` matches exactly, `-t NAME` prefix-matches.
- Queued jobs run against the filesystem as it will be later. Preflight them
  (the commit has the feature, the paths exist) and abort loudly. Log the tail
  of everything; never grep queued output down to the errors you expect.
- Verify sizes or digests after every transfer. Move data between boxes via
  Drive; direct box-to-box copies can crawl.
- Don't install packages into `/venv/main` while training runs from it.
- `wsparse` is an editable install, so a patched clone still imports the main
  checkout: launch clone code with `PYTHONPATH=<clone>/src` and check
  `python -c "import wsparse; print(wsparse.__file__)"`.
- Drive: count recursively (`rclone lsf -R`) before concluding something is
  gone; merge duplicate folders with `rclone dedupe --dedupe-mode newest` on the
  parent; use `--drive-chunk-size 128M` for big files.

## Tests, git, notes

- `python -m pytest tests/ -q` runs on CPU with synthetic data in a few
  minutes; run it before a campaign and after every pull. The 2-process DDP
  test is opt-in: `WSPARSE_DDP_TEST=1 python -m pytest tests/test_train_ddp.py -q`.
  Known flaky, deselect and report:
  `tests/test_interventions.py::test_alignment_summary_math_on_synthetic_rows`.
  Off a box there is usually no torch environment: use a throwaway venv
  (`pyyaml pytest torch numpy`) and `PYTHONPATH=src`.
- Git: commit and push straight to `main` (no branches, no PRs) after the suite passes.
- Research notes (`docs/*.tex`) follow `docs/rblapsum-kappa-stability-neutral.tex`:
  neutral tone, terms defined before use, measurements kept apart from
  interpretation, negative results stated plainly.
- Record new lessons about boxes or setup here or in the guide, not only in an
  agent's private memory.
