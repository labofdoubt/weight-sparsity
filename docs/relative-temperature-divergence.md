# Why relative-temperature soft Top(K+J) diverges (and absolute does not)

Case study: `uk_rout_soft_k32_j480_md` (MD/decouple, rope, k=32, j=480,
`temperature_scale_mode=relative`) — NaN by step 4000, watchdog-killed at 1600
with `feature_dead_frac` 0.95 — against `uk_rout_soft_k32_j480_md_abs`, which
differs in **exactly one config field** (`temperature_scale_mode=absolute`) and
trains fine (loss 2.18 @ 1600, dead_frac 0.0). Probes:
`probe_uk_rout_soft_k32_j480_md{,_abs}` (scores + both gradients, steps
0..1000 every 10, same seed and held-out batch).

**TL;DR — it is an implementation gap, with a clean theoretical reading.** The
LapSum surrogate's hand-written VJP is exact *for fixed (b, t)*; the barrier's
movement is accounted for, the temperature's is not. Under `relative` mode
`t = schedule · std(top-(k+j) scores)` is a function of the scores, and the
soft mask is then **exactly invariant to a common rescaling of a row's
scores** — so the true gradient has zero component along the score direction.
The implemented fixed-t gradient keeps that component: a **phantom** descent
direction the forward can never realize. Under absolute mode the same
component is *real* (the forward is not scale-invariant there) and comes with
a built-in brake: as scores grow in fixed-t units, every κ dies exponentially
and the surrogate lets go. Relative mode removes the brake — margins are
measured in units of the very scale that is inflating — so the phantom
compounds without opposition, per token, layer over layer (`residual_out`
placement). Scores are activations (`z = a ⊙ mask`), so score inflation is
activation inflation: loss *improves* until ~step 400, then the network
arithmetic collapses, features die, NaN.

## What the data shows

Reconstructed per-row barrier/temperature on the probe tensors (layer 7,
median over rows; z in units of t):

| step | rel: score p99.9 | rel: z_k | rel: κ-participation | abs: score p99.9 | abs: z_k | abs: κ-part |
|---|---|---|---|---|---|---|
| 0 | 5.0 | −0.90 | 176 | 5.0 | −1.00 | 205 |
| 100 | 62 | −1.95 | 398 | 41 | −0.47 | 71 |
| 300 | 1.3e3 | −1.31 | 228 | 18 | −0.68 | 92 |
| 500 | 4.3e5 | −1.18 | 205 | 12 | −0.81 | 120 |
| 1000 | 2.2e10 | −1.17 | 199 | 10 | −0.90 | 143 |

The relative run's t-relative geometry is **stationary while the scale runs
through ten decades** (that is the scale-free window: ~200 features keep
exchanging full-strength surrogate gradient forever). The absolute run has the
*same* early instability episode (~step 100, support collapse to ~70 features)
and **self-limits**: margins grow in absolute units, κ collapses, the surrogate
freezes, training pulls back. Training logs agree: rel effective `t` = 1.98 →
5.3 → 6.9e3 → 1.4e13 (steps 100→800) while its own `temperature_rel` diagnostic
reads 1.0 throughout; abs `temperature_rel` settles at 0.64–0.85.

Blown-row anatomy (rel, L7): the worst row *churns* (different tokens at
different probe steps; top-32 feature overlap 0–9/32 across steps), the **whole
512-wide candidate band inflates together** (the row's minimum band score
reaches 3e9; band std/median stays O(1–8)), the median row stays at ~50–100,
and `g_z` on already-blown rows is dead (1e-16) — inflation happens while a row
is still moderate, then compounds through the weights (under MD, it lands in
the softplus gains: the diverged arm ends with in_proj row gains at 73×).

## Hypotheses and verdicts

**H1 — algebra bug in the LapSum VJP.** *Refuted.* The implemented
`κ_i (u_i − ⟨q, u⟩)` equals the exact implicit-function derivative at fixed
(b, t): `f_i Σ q_j u_j ≡ q_i Σ f_j u_j` since `q = f/Σf`. The existing
finite-difference test (`test_vjp_matches_finite_differences_of_the_soft_mask`)
re-solves the barrier and passes. The gap is not *in* the formula; it is what
the formula treats as constant.

**H2 — scale-invariance removes the restoring force (theory).** *Confirmed by
measurement.* See the table: stationary z and participation under rel at 10
decades of scale growth; κ-collapse and frozen margins under abs. Necessary,
but on its own only says "nothing resists"; H3 supplies the push.

**H3 — the phantom scale-direction gradient is the engine (implementation).**
*Confirmed causally.* Patch: after the VJP, project out the score-direction
component per row (`g ← g − v̂⟨g, v̂⟩`, `v = r − mean(r)`; zero-sum makes the
centred projection equal the raw one and keeps the result zero-sum — verified
to 1e-15/1e-17 numerically). Result at step 1000, same config/seed
(`probe_gain_sweep`, 1000 steps):

| arm | CE @1000 | stream ‖x‖ L7 | gains L0/L7 |
|---|---|---|---|
| base (rel j480) | 5.71 | 2.0e15 | 72.9 / 8.0 |
| inactive_grad_scale=0 | 5.93 | 9.7e11 | 11.8 / 4.9 |
| **scale projection** | **2.077** | **70** | **0.99 / 1.17** |
| (abs twin's probe, reference) | 2.08 | — | — |

Removing exactly the component the forward cannot realize cures the divergence
outright and matches the absolute twin's CE to three digits. A live
instrumented probe (`analysis/diagnose_relative_temperature.py`) measures the
J-band scale projection `S_J = Σ (∂L/∂a · a)` per row along the reproduced
trajectory: a small positive inflation bias early (frac(S<0) up to 0.66 at
step 50), and at matched blown weights the absolute-mode counterfactual shows
**zero** inflation pressure (κ dead) while relative keeps renormalizing.

**H4 — the J band is the necessary carrier.** *Refuted as the root cause;
confirmed as the dose.* `inactive_grad_scale=0` (J band gets exactly zero
surrogate gradient) still diverges, three decades slower — the active band's
own phantom suffices. j sets the *rate*, through `t = std(top-(k+j))`: a wide
band full of sub-boundary scores keeps t large relative to the top gaps, so
the top-k stay inside the soft window (rel z_max ≈ 1.4–5.5) instead of
freezing out (abs z_max reaches 40). Dose–response at layer 7 (score p99.9 at
step 1000, relative runs): j=64 → 15.5, j=128 → 16.7, j=256 → 21.5 (still
NaN'd later, step 4000), j=480 → 2.2e10.

**H5 — feature death / usage collapse drives it.** *Refuted as cause.* Dead
fraction is ~0.03–0.05 while inflation is already exponential (steps 200–400)
and only reaches 0.93 after the loss has collapsed; the absolute twin passes
through the same early support-collapse episode (~70 features at step 100) and
heals to dead_frac 0.0. Death is downstream damage.

## Status and options

The scale projection is validated to step 1000; a full 20000-step validation
run is training as `uk_rout_soft_k32_j480_md_scaleproj` (patched clone at
`/workspace/exp/ws_scaleproj` on the uk box; patch inlined below). Options:

1. **Adopt the projection** -- now implemented on `main` as
   `activation_bottleneck.project_scale_gradient` (default `false`; requires
   `temperature_scale_mode="relative"`, and is rejected under absolute mode,
   where the scale component is genuine and must be kept). The projection runs
   inside `_LapSumProbs.backward`, before the `inactive_grad_scale`
   reweighting, adds ~6 elementwise band passes and no allocations, and
   measured +0.4% step time (within noise).
2. **Prefer absolute mode** for soft runs (the `_abs` sweep shows it trains
   well at every j tried, including j=480 and k=64/j=448 which collapse under
   relative).

Note the reach of this beyond j=480: every relative-mode soft run carries the
phantom (the dc `j128` late-training activation amplification studied in
`docs/activation-amplification.md` — where forcing absolute temperature cut
the tail 14.7× — is the same mechanism at a lower dose; this investigation
supplies the causal root it lacked).

The core of the patch (shipped form is flag-gated with a guarded denominator):

```python
# in _LapSumProbs.backward, after grad_scores = kappa * (grad_p - shared):
v = scores - scores.mean(-1, keepdim=True)      # scores saved in forward
v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
grad_scores = grad_scores - v * (grad_scores * v).sum(-1, keepdim=True)
```

Reproductions of the unstable run vary in blow-up timing (nondeterministic
kernels; same seed): the original probe hit 2.2e10 by step 1000, the phantom
reproduction 3.4e13 by 650, the gain-probe arm 2.0e15 by 1000 -- and a
mislaunched full-length rerun became an accidental fourth: launched from the
patched clone with `python -m wsparse.train`, it silently imported the
editable-installed *main* package (unpatched) and went NaN by step 2000 right
on schedule. Every relative reproduction diverges; both interventions and the
absolute twin never do. The proper full-length validation trains from `main`
with the shipped flag:

```bash
python -m wsparse.train --config /workspace/runs/uk_rout_soft_k32_j480_md/config.json \
    activation_bottleneck.project_scale_gradient=true \
    train.run_name=uk_rout_soft_k32_j480_md_scaleproj
```
