# TinyStories interpretability benchmark (v0)

Two questions, measured on one fixed bottleneck:

1. **Top-1 Sparse Probe** — does a known TinyStories concept correspond cleanly
   to a single bottleneck feature?
2. **Concept Localization** — when a concept is demonstrably present, does one
   bottleneck feature carry it, or is it spread across several?

No AutoInterp, no causal interventions, no layer sweep. The dataset is frozen
and shared by every model, so numbers are comparable across checkpoints.

## Running it

The dataset is built **once** and reused; do not regenerate it per model.

```bash
python interpretability/build_tinystories_concept_dataset.py \
    --data-dir data/tinystories --out benchmark_data

python interpretability/extract_bottleneck_activations.py \
    --checkpoint runs/<run>/latest.pt --data-dir data/tinystories \
    --benchmark benchmark_data --out activations/<run>.npz

python interpretability/sparse_probe.py \
    --activations activations/<run>.npz --benchmark benchmark_data \
    --out results/<run>

python interpretability/concept_localization.py \
    --activations activations/<run>.npz --benchmark benchmark_data \
    --results results/<run>

python interpretability/plotting.py --results results/<run>
```

Models with more than one bottleneck per block need `--placement` to say which
one to evaluate.

## Decisions worth knowing about

**The evaluation layer is fixed before any evaluation runs.** The installed
bottleneck nearest 50% depth (`n_layers // 2`), never chosen by score. Every
model uses the same relative layer. `evaluation_layer`, `bottleneck_location`,
`model_depth`, `bottleneck_width`, `K` are all recorded in each result file.

**Feature activations are the sparse coefficients that reach the decoder** —
the gate's output — not pre-TopK scores, encoder logits, gradients, or decoder
outputs.

**Signed bottlenecks are unfolded.** `abs_topk` and `gated_topk` keep signed
coefficients, and the two signs of one index need not mean the same thing, so
`m` features become `2m` virtual features (`z+`, `z-`). Reported feature labels
look like `1402+`. Configurable via `--signed`.

**Data is the held-out `val.bin` token stream**, recovered into stories by
splitting on EOS. Working from the tokens the model actually consumed (rather
than re-downloading TinyStories) means a target token index means the same
thing at build time and at extraction time. Stories are split 60/20/20 **by
story id, before any example is extracted**; `split_story_ids.json` records the
assignment and a digest of `val.bin`.

Byte-level reconstruction matters here: decoding tokens one at a time is wrong
for byte-level BPE, because a multi-byte character can straddle two tokens and
each half decodes to U+FFFD. That silently destroyed curly quotation marks —
which the dialogue concept is defined by. Stories carrying TinyStories' known
mojibake artefact (~5.7%) are dropped rather than mislabelled.

**Negatives come from the same coarse family as the positives**, so a feature
cannot win by encoding "is a noun". Every concept names its negative pool in
`concepts.yaml`; a word landing in both pools is ambiguous and is dropped from
that concept (this is what removes `bear` from `animal`, since it is also a
toy).

## Sanity checks

All run automatically and land in `aggregate_metrics.json`:

| check | expected | measured (`bn_pmlp_k32j64_rel`) |
| --- | --- | --- |
| shuffled labels | ~0.5 | 0.503 |
| random live feature | ~0.5 | 0.507 |
| top-1 minus random | clearly positive | +0.162 |
| story-id split disjointness | asserted | passes |
| feature selection on train only | by construction | — |
| decoder contribution identity | ~0 | 8e-08 |
| dead features never selected | by construction | — |

## A caveat about `BottleneckOutputProbeAccuracy`

Spec section 17 applies the residual probe direction `w_c`, fitted on `x`,
directly to the bottleneck output `x_hat`. That is the right test for a
sparse autoencoder, whose objective is `x_hat ~ x`.

**These bottlenecks are not autoencoders.** They are trained end to end with no
reconstruction term, so `x_hat` is a learned transformation of `x`, not an
estimate of it — measured `cos(x, x_hat) ~ 0`. Applying an `x`-space direction
to `x_hat` then reads as chance whether or not the concept survived, and
`ScatterRate` inherits the problem, since "survives" is defined through it.

Both are therefore reported:

* `BottleneckOutputProbeAccuracy`, `ScatterRate`, `RetainedElsewhere` — exactly
  as specified;
* `*_refit` — the same quantities with a probe fitted on `x_hat` itself, which
  asks "did the concept survive?" in the output's own coordinates.

On `bn_pmlp_k32j64_rel` the two disagree completely: 0.502 vs 0.923 output
probe accuracy, ScatterRate 0.117 vs 0.569. The specified version concludes the
bottleneck destroyed the concept; the refit shows it survives and is simply
carried elsewhere. **Use `_refit` whenever `reconstruction_cosine` is near
zero**, which is reported per concept.

## Outputs

```
benchmark_data/   concepts.yaml split_story_ids.json {train,validation,test}.parquet
                  qc_examples.html dataset_stats.json
results/<run>/    sparse_probe_per_concept.csv localization_per_concept.csv
                  aggregate_metrics.json selected_features.json
                  sparse_probe_topk_curve.png concept_localization.png
```

`selected_features.json` keeps the chosen feature id per concept (and the
fitted residual direction), so the same feature can be inspected by hand later.

---

The **bottleneck score / gradient / weight-norm** pipeline and its streamlit
viewer live in [`../analysis/`](../analysis/) with their own README. That is a
separate line of work from the concept benchmark above: it looks at the gate's
ranking scores, its gradients and the blocks' weight norms rather than at
concepts.
