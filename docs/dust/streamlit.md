# The DUST streamlit viewer

The DUST experiments reported in this folder have a streamlit viewer,
`dust_snr` (`analysis/dust_snr_viewer.py`), that can be launched on any box.
It shows the per-coordinate signal-to-noise ratio of the DUST estimate,
SNR_K = sqrt(K) |mu| / nu over the 1536 feature coordinates, on the K=32
code-carried model (`configs/cr_dust/snrprobe`: sigma 0.2, encoder target,
the full pool receiving the noise):

- tab "training populations": one model per training population (256, 1024,
  4096, 16384, 65536 draws; 20 steps, a checkpoint after every DUST step),
  with dropdowns for K and the population and sliders for the checkpoint,
  the bottleneck, the token position and the sample sequence; the histogram
  over coordinates, the token's summary numbers and the median per
  checkpoint;
- tab "single-block probes": one checkpoint of the population-4096 model
  perturbed in block 1 or block 6 only, sigma 0.2 or 1.0, 512 to 32768 draws;
  histogram of SNR_K, |mu| or nu with a bin-width slider, log or linear
  axis, and the pure-noise reference (|N(0, 1)| for SNR_K, a half-normal with
  scale nu / sqrt(K) for |mu|).

Its data (`analysis/dust_snr_probe.py` output) is on Drive:
`gdrive:weight-sparsity/analysis_paris/dust_snr` (6.2 GB) and
`gdrive:weight-sparsity/analysis_paris/dust_snr_single` (231 MB); the probed
runs are `gdrive:weight-sparsity/runs_paris/pa_snr_*`. No GPU is needed.

Launch on a box set up per AGENTS.md (repo, deps, rclone), from the repo
directory:

```bash
rclone copy gdrive:weight-sparsity/analysis_paris/dust_snr /workspace/analysis/dust_snr --drive-chunk-size 128M
rclone copy gdrive:weight-sparsity/analysis_paris/dust_snr_single /workspace/analysis/dust_snr_single
bash scripts/streamlit_install.sh install dust_snr
```

then tunnel `ssh -p <port> root@<host> -L 8501:localhost:8501` and open
http://localhost:8501. The full entry (service script, environment variables,
where it last ran) is in `docs/streamlits.md`, next to the two other viewers;
the MNIST toy's counterpart is `mnist_snr` (`toy_mnist/snr_viewer.py`, data
in the repo). The estimator and the SNR definitions are in `docs/dust-topk.md`.
