# toy_mnist: support optimization for a hard Top-K input gate on MNIST

A self-contained study folder (nothing here imports `wsparse`) of how
support optimization affects the training of a hard Top-K gate, in the
smallest setting that has one: a two-layer MLP on MNIST whose input passes
through a trainable diagonal rescaling and an AbsTopK gate.

## The model

With $x \in \mathbb R^{784}$ the flattened image (pixels in $[0, 1]$ under
`data.normalize: none`),

$$
u = e^{g} \odot x, \qquad
x' = \operatorname{AbsTopK}_K(u), \qquad
z = W_2\, \phi(W_1 x' + b_1) + b_2, \qquad \phi = \mathrm{ReLU},
$$

and the loss is the softmax cross-entropy of $z$.  Each stage is a config
switch:

| config | stage |
|---|---|
| `rescale.enabled` | the diagonal rescaling $e^{g} \odot x$; $g \in \mathbb R^{784}$ is trainable and starts at $0$ (the identity) |
| `topk.enabled`, `topk.k` | $\operatorname{AbsTopK}_K$: the $K$ entries of largest $\lvert u \rvert$ pass, the others are zero |
| `topk.j` | the pool $\operatorname{Top}_{K+J}(\lvert u \rvert)$ that a DUST step perturbs (the hard backward never reads it) |
| `topk.surrogate_mode` | `hard` or `dust` (below) |

The baseline `z = W_2 \phi(W_1 x)` is both switches off.

**Hard mode.** The mask is numerically hard and the backward passes it:
$\partial L / \partial u = m \odot \partial L / \partial x'$, so $g_n$
receives a gradient from a sample only when pixel $n$ is in that sample's
support.  A pixel that is never selected never moves (there is no weight
decay on $g$ or on the biases; `train.weight_decay` reaches $W_1$ and $W_2$
only).

## DUST mode

`topk.surrogate_mode: dust` keeps the hard gate and alternates ordinary hard
steps with DUST steps according to `dust_schedule = [n_hard, n_dust]`,
exactly as `surrogate_mode: dust` in the main repository
(`docs/dust-topk.md`, `src/wsparse/bottleneck/dust.py`).  The fields have the
same names, defaults and validation; `toy_mnist/dust.py` carries the formulas
adapted to one gate and no sequence axis.  Per batch of $B$ samples on a
DUST step:

1. The clean forward (train mode) records the gate input $u_t$ of every
   sample.  Its CE and accuracy are what the step logs as `train/ce` and
   `train/acc`.
2. For each sample, the clean pool $C_t = \operatorname{TopIndices}_{K+J}(\lvert u_t \rvert)$.
   Draw $i$ of `dust_population` $= K_d$ perturbs every sample at once,
   $$u'_t = u_t + \sigma_t\, a^{(i)}_t, \qquad a^{(i)}_t \sim \mathcal N(0, I)\ \text{on } C_t,\ 0 \text{ off it},$$
   on the signed value before the absolute value.  $\sigma_t$ is `dust_sigma`
   (`dust_sigma_mode: absolute`) or `dust_sigma` times the RMS of the clean
   pool values of the sample (`relative_rms`).
3. The tail (gate, MLP, CE) is rerun from $u'$ in eval mode under `no_grad`,
   `dust_chunk_draws` draws per forward, giving the per-sample CE
   $\ell^{(i)}_t$.
4. A sample has no later tokens, so the credit is its own loss,
   $R^{(i)}_t = \ell^{(i)}_t$ (`dust_gamma` must stay $0$); the baseline is
   the mean over the draws, $\tilde R_t = \frac 1{K_d} \sum_i R^{(i)}_t$.
5. The estimated output error, nonzero on $C_t$ only, divided by $B$ so that
   it estimates the gradient of the mean CE:
   $$\hat g_t = \frac{1}{K_d\, \sigma_t\, B} \sum_{i=1}^{K_d} \bigl(R^{(i)}_t - \tilde R_t\bigr)\, a^{(i)}_t .$$
6. `dust_target`.  The only parameter before the gate is $g$, with
   $\partial u_{t,n} / \partial g_n = e^{g_n} x_{t,n}$, so under `encoder`
   $$\hat G_g = \sum_t \hat g_t \odot e^{g} \odot x_t$$
   is added to `g.grad` in closed form (the diagonal of the outer product
   $\sum_t \hat g_t x_t^\top$ that the main repository forms for an encoder
   matrix, times $e^{g}$), and under `backprop` $\hat g$ is backpropagated by
   autograd from $u$, which reaches $g$ and nothing else.  The two targets
   give the same gradient here; both are kept for parity with the main
   repository and as a consistency check (the tests compare them).

`dust_grad_scale` multiplies $\hat g$ before either target;
`dust_optimizer: separate` applies the DUST gradient with a second AdamW
instance (own moments and step count) on DUST steps only; `dust_lr_scale`
multiplies the DUST steps' learning rate.  The noise comes from a dedicated
generator seeded from `train.seed`.  No parameter other than $g$ moves on a
DUST step.

What the estimate contains: for the smooth part of the loss it is (up to the
factor $1 - 1/K_d$ of the draw-mean baseline) the gradient of the
Gaussian-smoothed loss, and the selection changes a perturbation causes at
the gate are in the reward.  This is what the hard backward lacks: a
gradient for the $J$ pool entries outside the support, and for the support
entries' effect on the selection itself.

## Running

```bash
python -m toy_mnist.train --config toy_mnist/configs/topk_dust.yaml \
    train.run_name=my_run topk.dust_sigma=0.05
```

Overrides are `section.key=value` (lists as `topk.dust_schedule=[1,3]`).
Shipped configs (`toy_mnist/configs/`): `baseline`, `rescale`,
`topk_hard`, `topk_dust`, `topk_dust_backprop`; all include `base.yaml`
(hidden 256, AdamW lr $10^{-3}$, cosine over 3000 steps of batch 128, no
weight decay, no clipping).  MNIST is downloaded on first use into
`data.data_dir` (MD5-checked IDX files, cached as `mnist.npz`); the whole
set lives on the training device.

A run directory `train.out_dir/train.run_name` holds `config.json`,
`metrics.jsonl`, TensorBoard events (when `tensorboard` is installed),
`summary.json` (best and final validation, status) and `final.pt`.  A run
without `summary.json` did not finish.

Metrics: `train/ce`, `train/acc` (the clean hard forward on both kinds of
step), `train/lr`, `train/grad_norm` (the last step's total norm, whichever
kind), `val/ce`, `val/acc` (the 10 000 test images, deterministic),
`gate/features_seen` (input features active at least once in the window's
training forwards), `gate/usage_eff` ($\exp$ of the entropy of the
normalized usage EMA), `gate/usage_max`, `rescale/g_mean|std|min|max`, and
on DUST runs `dust/steps`, `dust/ce_perturbed`, `dust/ce_gap`,
`dust/reward_rms`, `dust/ghat_rms` (the unscaled estimate),
`dust/grad_norm`, `dust/lr`, `dust/cosine` (every `dust_cosine_every_steps`
DUST steps: the cosine between the DUST gradient of $g$ and the
hard-backward gradient of the clean CE for $g$).

## Tests

```bash
python -m pytest toy_mnist/tests -q
```

CPU, synthetic images, about a minute.  They check the gate's forward and
masked backward, the estimator's closed forms (and, when `wsparse` is
importable, bit-for-bit agreement with `wsparse.bottleneck.dust` on
one-token inputs), the two targets' equality, the alternation and which
parameters move on each kind of step, and the separate optimizer.

## Notes

- `AbsTopK` with pixels in $[0, 1]$ ranks by intensity; with
  `data.normalize: global` the background becomes a negative constant and
  the ranking is by distance from the mean intensity.
- `fc1` keeps PyTorch's default init (fan-in 784) although only $K$ inputs
  are nonzero after the gate; the pre-activation scale is reduced by about
  $\sqrt{K / 784}$ at initialization.
- Ties among equal $\lvert u \rvert$ (many background pixels are exactly 0)
  are broken by `torch.topk`, deterministic on a device but not across
  devices.
