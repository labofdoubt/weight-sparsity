# Why activations blow up in stream-placed bottlenecks

Working notes on an amplification mechanism spotted in the early-training
probes: at some `(block, sequence, position)` cells the bottleneck scores are
enormous — e.g. block 6, seq 2, pos 40 — and the effect is *stronger for hard
TopK with smaller k* and *far worse for soft Top(K+J) with j=128*, which is also
the run that destabilises later in training.

Status: mechanism identified for the compounding; the k- and j-dependence is
under test. Every number below is measured on the box, not estimated.

---

## 0. The observation, reproduced

`max|score|` at block 6, seq 2, pos 40, probe step 1000, and the same quantity
at every depth:

| run | k | j | max\|score\| | vs median cell | L0 → L7 growth |
| --- | --- | --- | --- | --- | --- |
| `hard_k64_selcorr` | 64 | – | 2.1e3 | 4.0x | **71x** |
| `hard_k32_selcorr` | 32 | – | 6.5e4 | 13.6x | **249x** |
| `soft_k32_j64` | 32 | 64 | 8.3e2 | 6.5x | **123x** |
| `soft_k32_j128` | 32 | 128 | 3.3e8 | **219,652x** | **54,910x** |

Two facts kill the obvious explanations straight away:

* **It is not a few outlier features.** In that cell 400–630 of the 1536
  features are within 10% of the largest, and `top1/top32` is only 2.4–3.2. The
  whole score vector is uniformly inflated — a *scale* effect, not the sparse
  "massive activations" phenomenon.
* **It is not a special token.** Position 40 of sequence 2 sits inside
  `" a time, there was a little brown dog named Spot."` — an ordinary word, not a
  delimiter or a sink token.

---

## 1. The driving force: per-layer gain > 1, compounded

Measured on `ckpt_step2000` of each run, per bottleneck, over a held-out batch
(`gain = ||x_hat|| / ||x||`, medians over 4x64 token positions):

| run | gain L0 | L3 | L7 | stream `\|\|x\|\|` L0 → L7 |
| --- | --- | --- | --- | --- |
| `hard_k32_selcorr` | 1.97 | 1.63 | 2.71 | 42.8 → 6007 (140x) |
| `soft_k32_j64` | 2.14 | 1.42 | 2.67 | 19.8 → 1220 (62x) |
| `soft_k32_j128` | 3.07 | 1.92 | 2.97 | 36.6 → 11600 (317x) |

**Every bottleneck at every depth has gain > 1.** Because `placement` is
`residual_out`, the bottleneck *replaces* the residual stream rather than sitting
in a branch, so nothing routes around it and the gains multiply:
`||x||` grows as `prod_l g_l`. This is the `g^n_layers` compounding the vast.ai
guide already warns about for stream placements — here caught in the act.

The gain at the flagged cell is *the same as the median* (2.11 vs 1.97 at L0 for
hard k32). So the cell is not amplified selectively. What makes it stand out is
that it starts about 2x above the median at L0 (84 vs 43) and then rides the
same uniform compounding — 2x at the input becomes 2x at the output of an
8-layer geometric chain, on top of a much larger absolute number.

### Why nothing stops it

`norm1` / `norm2` are RMSNorm, so attention and the MLP see a *scale-normalised*
copy of the stream and their outputs do not scale with `||x||`. The loss
therefore barely cares what the stream norm is — only `norm_f` at the very end
sees it, and it normalises too. **The absolute scale of the residual stream is
nearly free**, so there is no gradient pressure holding the bottleneck's gain at
1, and a systematic gain of ~2 per layer is not corrected.

A second-order consequence worth noting: since branch outputs do not grow with
`||x||`, the ratio `||branch|| / ||x||` decays as the stream inflates, so deeper
blocks contribute relatively less and the network drifts toward being dominated
by the bottleneck's own linear map.

### Where the gain comes from

Decomposing `gain = (||z||/||x||) x (||z_m||/||z||) x (||x_hat||/||z_m||)`:

| run | `\|\|z\|\|/\|\|x\|\|` @L0 | `\|\|z_m\|\|/\|\|z\|\|` @L0 |
| --- | --- | --- |
| `hard_k32` | 1.91 | 0.59 |
| `soft_j64` | 2.37 | 0.50 |
| `soft_j128` | **3.60** | 0.47 |

`||Win||F` (35–44) and `||Wout||F` (64–68) are *nearly identical across the three
runs*, so the difference between runs is not overall weight scale — it is how
much norm the encoder puts into `z` in the first place. That ordering
(1.91 → 2.37 → 3.60) tracks j exactly, which is the lead for §3.

At initialisation `init_mode=sqrt_k_selection_corrected` targets gain ≈ 1 (the
guide records `std(out)/std(in)` = 1.006 for it, against 0.128 for `default`).
So the gain of ~2 is **acquired during training**, not baked into the init.

---

## 2. CORRECTION: the first mechanism I proposed was wrong

Recorded in full because it was wrong in an instructive way, and because the
objection that killed it is a good test to apply to any story of this kind.

**What I claimed.** The surrogate mask is `m = m_hard + s*(p - p.detach())` on
output `z*m`, so `dL/dz_i = dL/dout_i * [m_hard_i + z_i*s*dp_i/dz_i]` carries a
term *proportional to z_i*. I called that multiplicative and therefore
exponential, and pointed at `g_z/g_ztilde` being exactly 1.000000 for hard and
~1.65 for soft as the evidence.

**The objection.** Gradients get *smaller* as activations grow, so a term
proportional to `z` cannot be what drives the growth.

**It is correct.** `dp/dz ~ 1/t` and `t ~ scale`, so `z_i/t` is O(1) --
scale-invariant, not growing -- and the absolute gradient falls 9 orders of
magnitude over the probe (2.9e-4 -> 2.2e-13). `g_z/g_ztilde ~ 1.65` only says
the surrogate adds a comparable-sized term; it says nothing about growth. I
inferred dynamics from algebra instead of measuring them.

Two further checks finished it off:

* `z = W_in x` is not a parameter, and the bottleneck weights barely move:
  `||W_in||F` is 39 at init and 35-44 at step 2000.
* Nor is a few blown-up rows hiding inside that Frobenius norm -- max/median
  **row** norm of `W_in` is 1.4-1.7 in every run, with **zero** rows above 10x
  the median. The bottleneck weights are innocent.

A salvageable observation: `|g_z| * |z|` is roughly conserved (5.9e-4 -> 3.1e-4
over the probe), which does mean the *relative* update stays O(1) and the growth
rate is near-constant (1.02-1.03 per step, sustained). But that product is
roughly conserved in **all four** runs (late/early 0.48-0.98), so it does not
discriminate healthy from runaway and cannot be the cause either.

---

## 3. The mechanism, as actually measured

### 3a. The spike is created by the MLP, not the bottleneck

Decomposing block 0's residual stream at step 2000 (`x = emb + attn + mlp`, the
bottleneck's input under `residual_out`):

| run | median `\|\|mlp\|\|` | worst `\|\|mlp\|\|` | ratio | worst `\|\|x\|\|` | `\|\|attn\|\|` ratio |
| --- | --- | --- | --- | --- | --- |
| `soft_j128` | 52.4 | **6.0e4** | **1147x** | 6.0e4 | 0.39 |
| `soft_j64` | 17.2 | 58.2 | 3.4x | 59.9 | 0.93 |
| `hard_k32` | 32.4 | 914 | 28x | 915 | 0.84 |

At the runaway token `||x||` equals `||mlp||` to four digits, and the attention
ratio is *below 1* -- attention is not involved. This is the familiar
massive-activation / outlier-feature behaviour of a transformer MLP, and it is
where the tail is born. It tracks the MLP weights: block-0 `fc2 ||W||F` is

| run | step 2000 | 12000 | 20000 |
| --- | --- | --- | --- |
| `hard_k32` | 31.8 | 61.9 | 60.3 |
| `soft_j64` | 31.0 | 57.4 | 56.2 |
| `soft_j128` | **115.5** | **453.7** | **560.2** |

roughly **9x larger** for j128 and still growing at step 20000.

### 3b. The bottleneck's role is to compound it, not create it

Following that same worst token down the stack (j128, step 2000):

| L | `\|\|x\|\|` | `\|\|mlp\|\|` | mlp/x | `x_l / x_{l-1}` |
| --- | --- | --- | --- | --- |
| 0 | 6.005e4 | 6.005e4 | **1.000** | – |
| 1 | 2.167e6 | 789 | 0.000 | **36.1** |
| 2 | 2.491e7 | 344 | 0.000 | 11.5 |
| 3 | 1.795e8 | 262 | 0.000 | 7.2 |
| 7 | 4.359e10 | 2561 | 0.000 | 3.5 |

The MLP makes the spike at block 0 and is then irrelevant (`mlp/x = 0.000`
everywhere after); the bottleneck multiplies it 36x, 11.5x, 7.2x ... to 4.4e10.
The same trace for `hard_k32` shows the same *shape* with much smaller numbers:
`||x||` 632 at L0 (mlp/x = 0.984) growing by only ~1.5-2.3x per block to 4.5e4.

**This also corrects a second claim of mine.** I earlier wrote that the gain is
not token-specific, based on the mild cell (2,40) where `gain@cell ~ median
gain`. For the genuinely extreme tokens it is emphatically token-specific: 36x
at L1 against a median gain of ~2. The bottleneck amplifies the outlier tokens
far harder than typical ones.

### 3c. Why nothing damps it

`norm1`/`norm2` are RMSNorm, so the branches see a scale-normalised copy of the
stream and the loss is nearly blind to the stream's absolute scale. `cos(x,
x_hat)` is ~0 at every layer and checkpoint, so the bottleneck never learns to
reconstruct -- it replaces the stream with a near-orthogonal vector, and there
is no residual path to dilute an outlier. Causally confirmed: the identical hard
k=32 config with `placement=pre_mlp` (a residual *branch*, so the skip routes
around the bottleneck) gives an 8-layer multiplier of **0.135** and L7
max|score| **6.03**, against **509** for `residual_out` -- and a better CE
(3.16 vs 5.27).

---

## 4. Q1 -- what is the driving force?

Two stages, in this order:

1. **The MLP emits an outlier** at a few token positions, growing with its `fc2`
   weight norm (§3a). This is not caused by the bottleneck; ordinary
   transformers do it too.
2. **The stream placement compounds it.** `residual_out` replaces the stream, so
   the outlier is multiplied by the bottleneck's gain at every later block, and
   the gain applied to outlier tokens is far above the median gain (§3b).

Neither stage is exponential-in-time on its own; the huge numbers are the
product of a large seed value and eight multiplications.

---

## 5. Q2 -- why does raising j make it worse?

**Established:** j128's block-0 MLP `fc2` norm is ~9x the other runs' and still
climbing (115 -> 560), which is what makes its seed spike ~100x larger (6.0e4 vs
622 for hard, 58 for j64), and its per-layer gain on that token is much larger
(36x at L1 vs ~2x). The controlled 300-step sweep confirms the monotone trend in
the global part: 8-layer multiplier 32.2 / 45.2 / 68.9 / 187.3 for j = 32 / 64 /
128 / 256, with L7 max|score| jumping 886 -> 2.20e4 between j=64 and j=128.

**Not established:** *why* a wider candidate pool drives the MLP's `fc2` weights
up. That is now the open question, and it is a question about the MLP, not about
the gate's algebra -- which is where I went wrong the first time.

One measured fact that may or may not be part of it: `t = schedule *
std(top-(k+j))` shrinks relative to the score scale as j grows -- `t/r[0]` =
0.161 / 0.147 / 0.134 / 0.120 for j = 32 / 64 / 128 / 256, measured on fixed
score vectors, so `kappa ~ 1/t` rises 1.20x from j=64 to j=128. And `t/scale` is
constant over the whole ladder, so the surrogate is scale-invariant and never
saturates. Whether that is causally upstream of the MLP growth is untested.

### The fix works, for a reason I cannot yet fully explain

Changing only `temperature_scale_mode` on j=128, 300 steps:

| | gain geo | stream x8 | L7 max\|score\| | CE |
| --- | --- | --- | --- | --- |
| j128 `relative` (default) | 1.697 | 68.9 | **2.20e4** | 4.264 |
| j128 **`absolute`** | 1.677 | 62.7 | **1496** | 4.933 |
| j64 `relative` (healthy, for scale) | 1.610 | 45.2 | 886 | 4.384 |

The tail falls **14.7x** while the global gain barely moves. That is a real,
reproducible effect and the most promising lever. But since the tail is now
known to originate in the MLP, the explanation I gave for *why* absolute
temperature helps (capping a multiplicative term in the gate) no longer stands.
Needs a run long enough to see whether the MLP `fc2` growth is suppressed too.

CE is worse at step 300 (4.933 vs 4.264). Worth noting -- though the relative
run's early CE advantage is exactly the pattern that later diverges (its CE
bottoms at step 14000 then climbs to 2.498), so a step-300 comparison settles
nothing about final quality.

---

## 6. Q3 -- why does raising k help hard TopK?

**Still open, and my first answer here was also wrong.** I proposed a static
`keep(k)` argument (`keep` = 0.385 at k=32 vs 0.495 at k=64, so `1.29^8 = 7.6x`
against a measured 10.3x). The arithmetic matched, so I believed it, but a
controlled 300-step sweep refutes it -- gain *rises* with k (1.31 / 1.46 / 1.64
at k = 32 / 64 / 128) and so does max|score|.

What the probe trajectories actually show is a difference in *timing* (L7 median
cell):

| step | 300 | 500 | 600 | 700 | 900 | 1000 |
| --- | --- | --- | --- | --- | --- | --- |
| `hard_k32` | 398 | 1652 | 4427 | 6670 | 1.62e4 | 1.53e4 |
| `hard_k64` | 116 | 1124 | **2277** | 1358 | 1640 | **1480** |

`hard_k64` peaks at step 600 and comes back down; `hard_k32` is still climbing
at step 1000, 10x higher. At step 300 -- where the sweep measured -- neither has
turned over, which is why the sweep ordering is reversed. So the question is
"why does a larger k turn the corner sooner", and a 1200-step sweep over the
same grid is running (E6). No k=64 checkpoint exists, so the MLP `fc2`
comparison that settled Q2 cannot yet be made for k.

---

## 7. What is established vs open

**Established (measured, with controls):**

* The spike originates in the **MLP** at a few tokens, not in the bottleneck
  (mlp/x = 1.000 at the seed block; attention ratio < 1).
* The bottleneck **compounds** it, and compounding requires the *stream*
  placement -- `pre_mlp` gives an 8-layer multiplier of 0.135 vs 509.
* The gain applied to outlier tokens is far larger than the median gain (36x vs
  ~2x), so outliers are amplified preferentially.
* Bottleneck weights are not the culprit: `||W_in||F` ~ constant, max/median row
  norm 1.4-1.7, zero blown-up rows.
* j128's block-0 MLP `fc2` norm is ~9x the others' and still growing.
* Absolute temperature cuts the tail 14.7x at fixed global gain.
* Global inflation rises monotonically with j (8-layer multiplier 32 -> 187 for
  j = 32 -> 256).

**Open:**

* Why a larger j drives the MLP's `fc2` weights up. (Q2's real remaining half.)
* Why a larger k makes a hard run turn over sooner. (Q3, E6 running.)
* Why absolute temperature suppresses the tail, now that the tail is known to be
  MLP-seeded.
* Whether the outlier tokens are the same ones across runs/seeds, and whether
  they are the tokens the model uses to carry information *through* a bottleneck
  that otherwise discards it (a plausible story -- a large enough value is
  guaranteed to survive TopK -- but untested).

**Retracted:** the `z_i`-proportional surrogate term as the driver (§2); the
claim that gain is not token-specific (§3b); the static `keep(k)` account of
Q3 (§6).
