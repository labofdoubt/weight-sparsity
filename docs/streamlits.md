# Streamlit viewers

Every streamlit app in the repo: what it shows, where its data lives, and how
to launch it exactly as it ran before. Boxes are temporary, so a viewer is
defined by the three things that outlive them: its script, its datasets (on
Drive, or committed in the repo) and its service script + supervisor conf in
`scripts/`. One entry per viewer; a new or moved viewer, or new data for one,
updates its entry in the same commit (AGENTS.md).

| viewer | shows | script | data | port |
|---|---|---|---|---|
| `score_explorer` | Top-K scores of the transformer bottlenecks over checkpoints and early training | `analysis/score_explorer.py` | Drive, `analysis_zurich/scores` (+ older prefixes) | 8501 |
| `dust_snr` | per-coordinate SNR of the DUST estimate on the transformer bottlenecks | `analysis/dust_snr_viewer.py` | Drive, `analysis_paris/dust_snr{,_single}` | 8501 |
| `mnist_snr` | per-coordinate SNR of the DUST estimate in the MNIST toy | `toy_mnist/snr_viewer.py` | in the repo, `toy_mnist/reports/snr_sweep` | 8502 |

## Launching a viewer on a box

After the box is set up per AGENTS.md (repo, deps, rclone), restore the
viewer's data as given in its entry, then from the repo directory:

```bash
bash scripts/streamlit_install.sh install <viewer>    # supervisor service, survives reboots
bash scripts/streamlit_install.sh status  <viewer>    # also: log, restart (after a git pull), stop
```

The services bind 127.0.0.1 only; reach one through a tunnel,
`ssh -p <port> root@<host> -L <viewer port>:localhost:<viewer port>` and
http://localhost:<viewer port> (or `-L 8080:localhost:<viewer port>` and
http://localhost:8080; the box's own 8080 is Jupyter). `score_explorer` and
`dust_snr` share port 8501: to serve both on one box, set
`STREAMLIT_SERVER_PORT` for one of them in its conf's `environment=` line
(`environment=PROC_NAME="%(program_name)s",STREAMLIT_SERVER_PORT="8503"`);
the service scripts take the port and the data roots from the environment and
fall back to the defaults listed below. A viewer can also run ad hoc with the
`streamlit run` line of its entry, inside tmux so it outlives the shell.

## score_explorer: bottleneck score explorer

Browses the Top-K ranking scores of the stream-placed bottlenecks of a run
across its checkpoints (steps 2k to 20k) or across the first 1000 steps of a
re-run with gradients, one dataset per run: rank versus |score| with the
K and K+J band edges, the half-line, the support map per feature index, the
raw score scale, and for code-carried runs the carry panels. Documentation:
`analysis/README.md` (section 10 for the service).

- Data roots (environment variable, default path): `SCORES_DIR`
  `/workspace/analysis/scores` (checkpoint ladders, `<run>.npy` +
  `<run>.json`, with `.code_residual`, `.g_z`, `.g_ztilde` sidecars for the
  code-carried runs), `PROBE_DIR` `/workspace/analysis/probe` (early-training
  probes), `WNORM_DIR` `/workspace/analysis/wnorm`, `SWAPS_DIR`
  `/workspace/analysis/swaps` (swap interventions; not supported for
  `code_residual` runs).
- On Drive, scores: `gdrive:weight-sparsity/analysis_zurich/scores` (56 GB,
  the state last served: the 2k-to-20k ladders of the 10 code-carried RBLapSum
  T=2 cells and their first-order variants, the 5 code-carried hard baselines,
  and the stream-carried kj-note runs, 32 datasets with sidecars);
  `analysis_lens_1/scores` (23 GB, the original lens_1 ladders) with
  `analysis_lens_1/probe` (238 GB) and `analysis_lens_1/wnorm`; older sets in
  `analysis_uk`, `analysis_france`, `analysis_taiwan_2` (the only `swaps/`)
  and `analysis_bergen`. `rclone lsf gdrive:weight-sparsity/<prefix>/scores`
  lists a prefix's datasets; the viewer reads whatever is in its roots.
- Restore, for example the last served state:
  `rclone copy gdrive:weight-sparsity/analysis_zurich/scores /workspace/analysis/scores --drive-chunk-size 128M`
  (and `analysis_lens_1/wnorm` to `/workspace/analysis/wnorm` for the weight-norm page).
- Service: `scripts/score_explorer_service.sh` + `scripts/score_explorer.supervisor.conf`, port 8501.
- Ad hoc: `SCORES_DIR=... PROBE_DIR=... WNORM_DIR=... streamlit run analysis/score_explorer.py`.
- Needs no GPU, corpus or checkpoints (the meta JSONs carry tokens, steps and gate config).
- Last served: zurich (1x 5090), 127.0.0.1:8501, 2026-10-08 to 2026-10-11; before that lens_1.

## dust_snr: DUST SNR per coordinate, transformer bottlenecks

Shows, for the DUST estimate on the K=32 code-carried model
(`configs/cr_dust/snrprobe`, sigma 0.2, encoder target, full pool), the
distribution over the 1536 feature coordinates of the per-coordinate
signal-to-noise ratio SNR_K = sqrt(K) |mu| / nu per checkpoint, bottleneck,
token and sample for the five training populations (256 to 65536, 20 steps,
a checkpoint after every DUST step), and in a second tab the single-block
probes of one checkpoint (block 1 or 6, sigma 0.2 or 1.0, 512 to 32768
draws) with the histogram of SNR_K, |mu| or nu, a bin-width slider, log or
linear axis and the pure-noise reference. Probe script:
`analysis/dust_snr_probe.py`; note `docs/dust-topk.md`.

- Data roots: `DUST_SNR_ROOT` `/workspace/analysis/dust_snr`
  (`<run>/snr_step<N>.npz`, SNR, |mu| and nu as float16 arrays of sample x
  layer x token x coordinate) and `DUST_SNR_SINGLE_ROOT`
  `/workspace/analysis/dust_snr_single` (`<run>/snr_step20_block<b>_sg<s>_K<K>.npz`).
- On Drive: `gdrive:weight-sparsity/analysis_paris/dust_snr` (6.2 GB, the
  five populations `pa_snr_k32_jall_sg0p2_p<P>`, 11 checkpoints each) and
  `gdrive:weight-sparsity/analysis_paris/dust_snr_single` (231 MB, 21 probes
  of `pa_snr_k32_jall_sg0p2_p4096`). The training runs and their checkpoints
  are in `gdrive:weight-sparsity/runs_paris/pa_snr_*`.
- Restore:
  `rclone copy gdrive:weight-sparsity/analysis_paris/dust_snr /workspace/analysis/dust_snr --drive-chunk-size 128M`
  and the same for `dust_snr_single`.
- Service: `scripts/dust_snr_service.sh` + `scripts/dust_snr.supervisor.conf`, port 8501.
- Ad hoc: `streamlit run analysis/dust_snr_viewer.py -- --root /workspace/analysis/dust_snr --single-root /workspace/analysis/dust_snr_single`.
- Needs no GPU; new probe files are picked up on the next page load.
- Last served: paris (8x 5090), 127.0.0.1:8501, supervisor program `dust_snr`, from 2026-10-23.

## mnist_snr: DUST SNR per coordinate, MNIST toy

Shows, for the `toy_mnist` AbsTopK input gate trained with DUST
(`toy_mnist/configs/snr_sweep`: K in {2, 4, 8, 16} x population in {128,
512, 2000, 8000}, plus K=2 at 32000, 32 checkpoints each), the distribution
over the 784 input coordinates of SNR_n = sqrt(P) |mu_n| / nu_n at a chosen
checkpoint (histogram on a log axis, fixed across the checkpoint slider),
the same values on the 28 x 28 pixel grid, and the median with its
interquartile band over the checkpoints; dropdowns pick K and P. Probe
script: `toy_mnist/snr_probe.py`; documentation `toy_mnist/README.md`.

- Data root: `SNR_ROOT`, default `/workspace/weight-sparsity/toy_mnist/reports/snr_sweep`
  (`<run>/snr.npz`, one row per checkpoint, `snr_u`/`snr_g` with mu and nu).
- In the repo: the 17 `snr.npz` of the 2026-10-23 sweep are committed under
  `toy_mnist/reports/snr_sweep/` (13 MB), so a checkout is all the viewer
  needs. The runs themselves (checkpoints, metrics) are in
  `gdrive:weight-sparsity/runs_osaka_mnist/snr_sweep`.
- Service: `scripts/mnist_snr_service.sh` + `scripts/mnist_snr.supervisor.conf`, port 8502.
- Ad hoc: `streamlit run toy_mnist/snr_viewer.py --server.port 8502 -- --root toy_mnist/reports/snr_sweep`.
- Needs no GPU and nothing outside the checkout.
- Last served: paris, 127.0.0.1:8502, ad hoc in tmux `mnist_snr_viewer` from
  a second checkout `/workspace/mnist_viewer/weight-sparsity`, 2026-10-23;
  before that osaka_mnist on 8501.
