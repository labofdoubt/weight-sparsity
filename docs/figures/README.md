# Figures

One subfolder per campaign/note; every new figure goes into a
subfolder (create one for a new campaign), never into this root.

- `depth/` — 24-layer stream-bottleneck depth study (stream-bottleneck-depth*.tex,
  `scripts/plot_depth.py`; `data/` holds the probe JSONs and curves)
- `diag500m/` — 500M-scale diagnostics, in two subfolders: `layer_diag/`
  (activation/gradient probes per layer, `analysis/layer_diag.py`) and
  `module_gain/` (forward/backward gain of the bottleneck and its post-norm,
  and its K sweep, `analysis/module_gain.py`)
- `init-pi/` — init-Pi heatmap grids (init-pi-grid*.pdf)
- `kappa-ablation/` — kappa ablation: mass, gradient, encoder-row measurements (rblapsum-kappa-ablation.tex)
- `kj/` — Top-(K+J) vs hard Top-K' (kj-vs-hard-topk.tex)
- `kstab/` — kappa stability campaign + divergence forensics (rblapsum-kappa-stability*.tex)
- `mdinit/` — MD-init vs decoupling campaign (docs/md-init-vs-decoupling.tex)
- `rho/` — signal-permutation ablation (rblapsum-signal-permutation.tex)
- `scale-dynamics/` — scale-dynamics notes: drift/heating, Pi fate maps, simplified gain (scale-dynamics-note*.tex)
- `sf/` — soft-forward campaign (rblapsum-soft-forward.tex)
- `sfv/` — support-only value gradient campaign (rblapsum-sf-support-values.tex)
- `stabilization/` — five-arm stabilization comparison (rblapsum-stabilization.tex)
- `surr/` — RBLapSum in the 24-layer code-residual stack (rblapsum-code-residual*.tex,
  `scripts/plot_surr.py`; `data/` holds the probe JSONs, onset series and curves)
