"""Configuration objects for model / data / training / bottleneck.

Configs are plain dataclasses.  They can be built from YAML files (with an
optional ``_base_`` key for composition) and overridden from the command line
with dotted ``--section.field=value`` flags, e.g.::

    python -m wsparse.train --config configs/ltp_base.yaml \
        --train.lr=6e-4 --activation_bottleneck.k=32
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_type_hints

import yaml


# Embedding tables are initialized at this std unless overridden, so that
# tok_emb + pos_emb has unit variance per element at init: 2 * (1/sqrt(2))**2 = 1.
DEFAULT_STD_EMBEDDING: float = 1.0 / math.sqrt(2.0)

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Decoder-only transformer.

    Defaults: no biases anywhere, RMSNorm, learnable (absolute) positional
    embeddings (``pos_encoding="rope"`` switches to rotary), pre-norm residual
    blocks.
    """

    vocab_size: int = 50257  # overwritten by the tokenizer at build time
    max_seq_len: int = 512

    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    mlp_ratio: float = 4.0  # d_mlp = round(mlp_ratio * d_model), rounded to a multiple of 8
    mlp_activation: str = "gelu"  # gelu | relu | silu | swiglu

    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = False  # biases in linear layers (off by default)
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    # ---- positional encoding ---------------------------------------------- #
    # "learned": an absolute nn.Embedding(max_seq_len, d_model) added to the
    #     token embedding.  The historical default -- every checkpoint saved
    #     before this field existed used it, and config_from_dict fills the
    #     default in for them, so they keep loading unchanged.
    # "rope": rotary position embeddings applied to q and k inside every
    #     attention layer (NeoX/Llama rotate-half convention).  No position
    #     table exists at all: init_std_pos is unused, the parameter count drops
    #     by max_seq_len * d_model, and block 0's input is the token embedding
    #     alone -- so the "tok + pos has unit variance" reasoning behind
    #     DEFAULT_STD_EMBEDDING does not apply to this mode.
    pos_encoding: str = "learned"  # learned | rope
    rope_theta: float = 10000.0  # base of the rotary frequency geometric series

    # ---- magnitude-direction decoupling (arXiv:2606.25971) ----------------- #
    # When true, weights are optimized as a fixed-Frobenius-norm *direction*
    # times learnable per-row/per-column softplus gains, embeddings and the LM
    # head are held at unit L2 row norm, and the input embedding is upscaled by
    # a fixed sqrt(d_model) in the forward.  The model still stores ordinary
    # fused weight tensors -- the split lives inside the optimizer step -- so
    # checkpoints stay plain.  This OVERRIDES every other init field
    # (init_scheme / init_std / init_gain / init_std_embedding /
    # init_scale_residual, and the bottleneck's init_mode), disables weight
    # decay for all norm-constrained parameters, and requires pos_encoding=
    # "rope" (the method defines no treatment for a learned position table).
    decouple: bool = False
    # "row_col": every matrix gets both gains (the paper's default, best in
    # their ablation).  "up_down": one-sided -- d_out >= d_in gets the row gain,
    # d_out < d_in the column gain (nGPT-style alternation).
    decouple_gains: str = "row_col"  # row_col | up_down
    # ---- bottleneck direction geometry (MD path only) --------------------- #
    # How ``md_init_`` draws the bottleneck's two projections.  Everything else
    # in the model -- attention, MLP, embeddings, the residual scaling -- is
    # untouched by this field.
    #   "standard"    what MD has always done: entrywise N(0, 1/d_model) then
    #                 projected to the c_F sphere, so the directions are random
    #                 Gaussian and the singular values spread over a
    #                 Marchenko-Pastur bulk.
    #   "orthogonal"  the encoder rows and the decoder columns are drawn as two
    #                 INDEPENDENT tight frames instead: W_hat_E^T W_hat_E =
    #                 (d_b/d) I and W_hat_D W_hat_D^T = (d_b/d) I, i.e. every
    #                 singular value equals sqrt(d_b/d).  Same Frobenius norm
    #                 (sqrt(d_b)), so the MD sphere radius c_F is unchanged, and
    #                 the same mean encoder-row / decoder-column norm of 1 --
    #                 only the *shape* of the spectrum differs.  Requires the MD
    #                 path (decouple or md_init) and an untied decoder.
    bottleneck_init: str = "standard"  # standard | orthogonal
    # A fixed, NON-learnable global scale on the bottleneck decoder, carried as
    # part of the parameterization
    #     W_D = g_D * diag(g_row) W_hat_D diag(g_col),
    # applied inside the forward graph, so autograd puts the same g_D factor
    # into the gradients of W_hat, g_row and g_col (see SparseTopKBottleneck).
    #   "none"                 g_D = 1
    #   "backward_preserving"  g_D = sqrt(d_model / K_eff), the scale at which
    #                          an isotropic gradient crosses the bottleneck with
    #                          unit energy; K_eff comes from
    #                          bottleneck.effective_backward_support (K for the
    #                          hard forward, K+J when the surrogate also moves
    #                          the J candidates).
    # Deliberately kept out of the row/column gains: those stay at 1 so the
    # semantics stay clean (direction / per-row / per-column / global).
    bottleneck_decoder_scale: str = "none"  # none | backward_preserving

    # The MD *initialization* without the MD *optimizer*: apply exactly the
    # same re-initialization decouple=True applies (md_init_: unit-L2 embedding
    # rows with the fixed sqrt(d_model) forward upscale, every other >=2-D
    # weight entrywise N(0, 1/d_model) projected to c_F in Frobenius norm,
    # biases zeroed), then train with the ordinary AdamW: no per-row/column
    # gains, no projection back to the sphere after the step, and
    # train.weight_decay applies as configured.  At the same seed the initial
    # weights are bitwise identical to a decouple=True run, so the pair
    # isolates what the norm constraint itself contributes.  Like decouple it
    # OVERRIDES every other init field and requires pos_encoding="rope" and
    # logit_scale="none" (the "auto" multiplier is derived from init fields
    # this mode overrides).  Mutually exclusive with decouple=True.
    md_init: bool = False

    # ---- initialization -------------------------------------------------- #
    # "fixed_std": every weight ~ N(0, init_std**2)
    # "fan_in":    every weight ~ N(0, (init_gain**2) / fan_in)
    init_scheme: str = "fixed_std"
    init_std: float = 0.02
    init_gain: float = 1.0
    # embeddings ignore init_scheme; they use these stds, which default to
    # DEFAULT_STD_EMBEDDING (1/sqrt(2)) rather than to init_std
    # The embedding (encoder) std.  Under tie_embeddings this is the single
    # shared matrix, so it sets the unembedding too.
    init_std_embedding: Optional[float] = None  # defaults to DEFAULT_STD_EMBEDDING
    init_std_pos: Optional[float] = None  # defaults to init_std_embedding
    # The unembedding (decoder) std, used only when the two are untied -- tied,
    # there is one matrix and init_std_embedding sets it.  Deliberately
    # independent of init_scheme/init_std/init_gain: lm_head maps a unit-RMS
    # residual to logits rather than feeding another layer, so it is scaled from
    # that requirement, not from the linear-layer convention.  Defaults to
    # 1/sqrt(d_model), which puts the init logits at unit std.
    init_std_unembedding: Optional[float] = None
    # scale the init of every residual-output projection by 1/sqrt(2 * n_layers)
    init_scale_residual: bool = True

    # ---- output scaling -------------------------------------------------- #
    # The head reads a unit-RMS residual, so the init logits land at std
    # unemb_std * sqrt(d_model).  "auto" divides that out, normalizing them to
    # ~unit std whatever the head's own scale is; at the default
    # init_std_unembedding it is already 1, so the factor is a no-op and only a
    # deliberately-scaled head (tied, or an explicit init_std_unembedding) is
    # actually rescaled.  "none" leaves the logits as lm_head produces them.
    # Caveat for tied heads: the head is the token embedding, which the residual
    # stream still carries, so unit-std logits sharpen a readout of the *current*
    # token and start the next-token loss above ln(vocab).
    logit_scale: str = "auto"  # auto | none

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.mlp_activation not in ("gelu", "relu", "silu", "swiglu"):
            raise ValueError(f"unknown mlp_activation: {self.mlp_activation}")
        if self.init_scheme not in ("fixed_std", "fan_in"):
            raise ValueError(f"unknown init_scheme: {self.init_scheme}")
        if self.logit_scale not in ("auto", "none"):
            raise ValueError(f"unknown logit_scale: {self.logit_scale}")
        if self.decouple_gains not in ("row_col", "up_down"):
            raise ValueError(
                f"unknown decouple_gains: {self.decouple_gains!r} (row_col | up_down)")
        if self.decouple and self.pos_encoding != "rope":
            raise ValueError(
                "decouple=True requires pos_encoding='rope': magnitude-direction "
                "decoupling defines no treatment for a learned position table "
                "(the paper trains with rope)")
        if self.decouple and self.logit_scale == "auto":
            raise ValueError(
                "decouple=True needs logit_scale='none': the 'auto' multiplier is "
                "derived from init fields that decoupling overrides, and unit-norm "
                "head rows reading a unit-RMS stream give unit-scale logits already")
        if self.md_init and self.decouple:
            raise ValueError(
                "md_init=True is redundant under decouple=True: decouple already "
                "applies the MD initialization (and the norm-constrained "
                "optimizer); set exactly one of the two")
        if self.md_init and self.pos_encoding != "rope":
            raise ValueError(
                "md_init=True requires pos_encoding='rope': the MD initialization "
                "defines no treatment for a learned position table")
        if self.bottleneck_init not in ("standard", "orthogonal"):
            raise ValueError(
                f"unknown bottleneck_init: {self.bottleneck_init!r} "
                "(standard | orthogonal)")
        if self.bottleneck_decoder_scale not in ("none", "backward_preserving"):
            raise ValueError(
                f"unknown bottleneck_decoder_scale: "
                f"{self.bottleneck_decoder_scale!r} (none | backward_preserving)")
        md_path = self.decouple or self.md_init
        if self.bottleneck_init == "orthogonal" and not md_path:
            raise ValueError(
                "bottleneck_init='orthogonal' is implemented on the MD path; "
                "set model.decouple=true or model.md_init=true (the "
                "non-decoupled bottleneck keeps activation_bottleneck.init_mode)")
        if self.bottleneck_decoder_scale != "none" and not md_path:
            raise ValueError(
                "bottleneck_decoder_scale is applied by the MD initialization; "
                "set model.decouple=true or model.md_init=true")
        if self.md_init and self.logit_scale == "auto":
            raise ValueError(
                "md_init=True needs logit_scale='none': the 'auto' multiplier is "
                "derived from init fields that the MD initialization overrides")
        if self.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"unknown pos_encoding: {self.pos_encoding!r} (learned | rope)")
        if self.pos_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"rope rotates pairs of head dimensions, so head_dim must be even; "
                f"got d_model={self.d_model} / n_heads={self.n_heads} = {self.head_dim}"
            )
        if self.tie_embeddings and self.init_std_unembedding is not None:
            raise ValueError(
                "init_std_unembedding is meaningless with tie_embeddings=True: "
                "there is one matrix, set init_std_embedding instead"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_mlp(self) -> int:
        raw = int(round(self.mlp_ratio * self.d_model))
        return max(8, (raw + 7) // 8 * 8)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    data_dir: str = "data/tinystories"
    seq_len: int = 512

    # "gpt_neo"  -> EleutherAI/gpt-neo-125M tokenizer (the one used by the
    #               original TinyStories models; vocab 50257)
    # "bpe"      -> a small byte-level BPE trained on TinyStories itself
    tokenizer: str = "gpt_neo"
    tokenizer_path: str = "data/tinystories/tokenizer"  # for tokenizer == "bpe"
    bpe_vocab_size: int = 8192
    bpe_train_docs: int = 200_000

    num_proc: int = 8
    val_fraction: float = 0.0  # 0 -> use the dataset's own validation split


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    # ---- optimizer ------------------------------------------------------- #
    optimizer: str = "adamw"
    lr: float = 6e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # weight decay is only applied to >= 2D parameters (matmul weights);
    # norm gains and biases are excluded.

    # ---- lr schedule ----------------------------------------------------- #
    lr_schedule: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1  # final lr = min_lr_ratio * lr

    # ---- batching -------------------------------------------------------- #
    batch_size: int = 32  # sequences per optimizer step (per device, after accumulation)
    # One extra evaluation over this many batches after the last step (0 = off).
    # Use it to score the FULL validation holdout once, while val_batches keeps
    # the routine (every validate_every_steps) evaluation on a small fixed
    # prefix of the same holdout: evaluate() reads deterministic disjoint
    # windows from the start of val.bin, so val_batches IS the size of the
    # routine subset.  Logged as val_final/ce, recorded in summary.json.
    final_val_batches: int = 0
    micro_batch_size: Optional[int] = None  # None -> == batch_size (no accumulation)

    # ---- length / cadence ------------------------------------------------ #
    max_steps: int = 20_000
    log_every_steps: int = 20
    validate_every_steps: int = 500
    val_batches: int = 40
    checkpoint_every_steps: int = 2000
    keep_last_checkpoints: int = 3
    sample_every_steps: int = 0  # 0 disables periodic sampling
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 96
    sample_count: int = 4  # generations per sampling event, drawn as one batch

    # ---- runtime --------------------------------------------------------- #
    seed: int = 1337
    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    compile: bool = False
    out_dir: str = "runs"
    run_name: str = "run"
    resume: str = ""  # path to a checkpoint, or "auto" to pick up the latest
    wandb_project: str = ""  # empty disables wandb
    wandb_entity: str = ""
    # TensorBoard events under <out_dir>/<run_name>/tb; point tensorboard at
    # <out_dir> to overlay every run in one chart
    tensorboard: bool = True

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        if self.micro_batch_size is None:
            self.micro_batch_size = self.batch_size
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"micro_batch_size={self.micro_batch_size}"
            )
        if self.lr_schedule not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown lr_schedule: {self.lr_schedule}")

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_size // int(self.micro_batch_size)



# --------------------------------------------------------------------------- #
# activation bottleneck
# --------------------------------------------------------------------------- #


@dataclass
class ActivationBottleneckConfig:
    """A hard-TopK activation bottleneck inserted in front of selected MLPs.

    ``W_in`` and ``W_out`` are dense and trained normally; what is sparse is
    the code between them.

        x_mlp -> W_in -> TopK/AbsTopK (exactly k of n_features) -> W_out -> MLP

    The forward pass is exact hard TopK.  The backward pass additionally lets
    the next ``j`` candidates move, through a Laplace-CDF soft mask at the
    constant ``temperature`` below; ``surrogate_mode`` picks the variant and
    with it what the backward does at the surrogate's boundary exchange.
    """

    enabled: bool = False

    layers: Any = "all"  # all | even | odd | first:n | last:n | [0, 2, 4]
    # pre_mlp     : on the MLP input, inside the residual branch -- the skip
    #               routes around it, so x still carries what was dropped.
    # residual     : on the stream itself, at the head of the block (before
    #               attention).  Nothing routes around it.
    # residual_out : on the stream itself, at the tail of the block (after the
    #               MLP add).  With layers="all" this differs from `residual`
    #               in one position only: it bottlenecks the final hidden state
    #               feeding the unembedding instead of the embedding output.
    # post_attn    : on the attention output, before the residual add.
    # post_mlp     : on the MLP output, before the residual add.  These two are
    #               also inside a branch, but constrain what the branch may
    #               *contribute* rather than what it may read.
    # Several may be combined ("post_mlp,post_attn"); each installs its own
    # bottleneck, so the parameter cost scales with how many are named.
    # Typed Any, not str, for the same reason as `layers`: the CLI turns a bare
    # "a,b" into a list, and a declared str would coerce that to its repr.
    placement: Any = "pre_mlp"  # pre_mlp | residual | residual_out | post_attn | post_mlp

    n_features: int = 4096  # N
    k: int = 256  # active in the forward pass
    j: int = 768  # extra candidates that only receive gradient

    # topk / abs_topk rank by the single projection's output (or its
    # magnitude).  gated_topk adds an independent score projection: the support
    # is ranked by s, the value v is carried separately, so dL/dv is the exact
    # hard-mask gradient while dL/ds is the constrained LapSum VJP.
    selection_mode: str = "abs_topk"  # topk | abs_topk | gated_topk

    # lapsum      -> soft-in-gradient LapSum: hard TopK forward, exact
    #                fixed-temperature barrier VJP (sum p_i = K) in the backward
    # rblapsum    -> hard forward, rank-boundary local kernel (see below)
    # rblapsum_sf -> the probabilities in the forward: y_i = z_i p_i
    # hard        -> plain hard-mask backward, no surrogate
    surrogate_mode: str = "hard"

    # Constant kernel/barrier temperature shared by the lapsum and rblapsum
    # modes (unified 2026-09-28; formerly fixed_temperature and
    # rblapsum_temperature).  Deliberately not score-scaled.
    temperature: float = 1.0


    # ---- rblapsum (surrogate_mode: rblapsum | rblapsum_sf) ------------------ #
    # Rank-Boundary LapSum: LapSum's hard TopK forward, Top(K+J) candidates and
    # exponential local kernel, but the boundary is the HARD rank b=max(b0,
    # s_(K+1)) instead of the soft mass constraint sum p_i = K -- an upper cap
    # of K active features, not exactly K.  No target-count penalty.
    #
    # rblapsum_sf ("soft forward") keeps the same pool, boundary and kernel but
    # puts the probabilities IN the forward: every Top(K+J) candidate outputs
    # z_i * p_i with p_i = F((s_i - b)/T) (Laplace CDF), features outside the
    # pool output exactly 0, and the backward is the gradient of that forward
    # -- no train-time forward/backward discrepancy.  boundary_grad_mode picks
    # how the boundary's own derivative is routed (through_rank at scale 1 is
    # the exact autograd gradient, since b IS the (K+1)-st score; the kappa
    # mode redistributes the same total kappa-weighted); rblapsum_support_scale
    # scales ONLY that boundary term in this mode.  In eval the gate falls back
    # to the hard Top-K forward while hard_inference=True (the default), so
    # val/ce stays comparable across regimes; the soft-forward CE is logged
    # separately as val_soft/ce.
    #   detach        independent local boundary gradients (default)
    #   project       + remove the common-mode score direction (cap-active only)
    #   through_rank  differentiate through the (K+1)-st score used as boundary
    #                 CAUTION: at sharp T the point-mass compensation on the
    #                 boundary feature leaves the rest of the pool per-feature
    #                 uncompensated and the boundary score can run away (j32/T1
    #                 diverged ~1.6k steps, both seeds)
    #   through_rank_kappa  the same zero-sum correction distributed
    #                 kappa-weighted -- LapSum's rank-one Jacobian form at the
    #                 rank boundary; removes the runaway (verified same-seed)
    # None -> mode-dependent default, resolved in __post_init__: "detach" for
    # the hard-forward rblapsum (the original default), "through_rank_kappa"
    # for rblapsum_sf (the soft forward's natural companion).  Set explicitly
    # for the ablations.
    rblapsum_boundary_grad_mode: Optional[str] = None
    # Multiplies the rblapsum SUPPORT gradient -- the boundary-exchange term
    # g_s -- in the backward, after the mode correction, so the zero-sum
    # structure of through_rank_kappa is preserved and the term is uniformly
    # rescaled rather than reshaped.  The hard task path (upstream * mask) is
    # untouched, so 0.0 is exactly the hard-TopK backward and 1.0 is the
    # unmodified surrogate.  Previously an analysis-only attribute set by
    # analysis/scale_dynamics.py; as a config field it can be trained with.
    # Under rblapsum_sf the whole score gradient is a true forward gradient,
    # so this scales ONLY the boundary term (- dist * sum(a)): 1.0 is the full
    # boundary derivative, 0.0 reproduces boundary_grad_mode="detach" -- the
    # direct z*kappa path is never scaled.  rblapsum modes only.
    rblapsum_support_scale: float = 1.0

    # Permutation ablation of the surrogate signal (surrogate_mode: rblapsum
    # only).  Each training backward, for every (token, block) row
    # independently, a rho fraction of the Top(K+J) pool is sampled and the
    # dL/dp_i = u_i z_i values AT those positions are shuffled by a uniform
    # random permutation of the subset -- BEFORE the kernel weighting, so each
    # position keeps its own kappa and the zero-sum correction is applied to
    # the permuted signal.  Preserves the surrogate's scale profile, kernel
    # locality, zero-sum structure and the per-row multiset of signal values;
    # destroys only the assignment of signal to neuron.  The hard task path
    # is never permuted.  0.0 (default) is exactly the unmodified backward;
    # 1.0 permutes the whole pool.  Training only; eval is unaffected.
    rblapsum_rho_random_perm_prob_grad: float = 0.0

    # rblapsum_sf only: which candidates' VALUES train.  "pool" (default) is
    # the gradient of the soft forward -- every Top(K+J) member gets the value
    # path u * p_i.  "support" masks the value path to the hard support: the J
    # inactive candidates receive only the score-path (support-change) term,
    # so their ranking still trains but their content does not.  The forward
    # is unchanged, so "support" deliberately reintroduces a forward/backward
    # discrepancy on the tail; active features keep u * p (not u * 1).
    rblapsum_sf_value_grad: str = "pool"

    # An RMSNorm on each bottleneck's own output, inside the module (so it is
    # part of the bottleneck's state_dict and the gate hooks are unaffected).
    # Motivation: under a stream placement the bottleneck's output scale
    # compounds through depth, so at fixed K the scores a deep block ranks can
    # be much larger than an early block's; normalizing the output pins that
    # scale per block.  The bottleneck whose output already feeds a norm --
    # residual_out in the final block, which feeds norm_f -- does not get one.
    post_norm: bool = False

    # b0: a FIXED bottleneck-level activation floor (never data-dependent).
    # For abs_topk, b0=0 makes almost every feature eligible, so L0 stays ~K; a
    # positive b0 is needed to get L0 < K on some tokens.  Strict s > b0.
    # None -> 0.0: no floor.  (Earlier campaigns ran abs_topk with a 0.1
    # default; measurements on those runs showed the floor essentially never
    # binds -- mean boundary 8-40x above it -- so the default is now the
    # neutral 0.0.  Set explicitly to reproduce the old behaviour.)
    # Resolved to a concrete float in __post_init__.
    rblapsum_boundary_floor: Optional[float] = None

    # How the bottleneck's own projections are initialized.  They are spliced in
    # after the model's _init_weights pass, so "default" means PyTorch's
    # nn.Linear defaults -- U(+-1/sqrt(fan_in)) with random biases -- which is
    # what every run before this option used.
    #   sqrt_k               encoder std 1/sqrt(d_model), decoder std 1/sqrt(k).
    #                        The decoder's fan-in is n_features but only k
    #                        coefficients are non-zero, so scaling by n_features
    #                        under-scales the output by sqrt(n/k).  Correcting
    #                        to k overshoots, since the survivors are the
    #                        largest coefficients rather than typical ones.
    #   sqrt_k_selection_corrected
    #                        sqrt_k divided by E[|z| | selected], the mean
    #                        magnitude of a surviving coefficient.  Lands at
    #                        unit output scale.
    #   unit_norm_dictionary both std 1/sqrt(d_model): decoder columns have
    #                        expected unit norm, the usual sparse-coding
    #                        convention.
    # default | sqrt_k | sqrt_k_selection_corrected | unit_norm_dictionary
    init_mode: str = "default"
    # Share one matrix between encoder and decoder (decoder = encoder^T).  Only
    # meaningful when both sides have the same scale, so it requires
    # unit_norm_dictionary.  Halves the bottleneck's parameters.
    tie_encoder_decoder: bool = False
    # One encoder and one decoder for EVERY installed bottleneck -- all selected
    # layers and, under a combined placement, all placements -- instead of a
    # fresh pair per module.  in_proj (plus score_proj under gated_topk) and
    # out_proj are then the same nn.Linear objects in every bottleneck: the
    # parameter cost is one bottleneck's worth however many are installed, and
    # each matrix receives the sum of every bottleneck's gradient.  The
    # state_dict still lists the matrices under every bottleneck's prefix (all
    # copies identical, as with a tied lm_head), so checkpoints load either
    # way.  What stays per bottleneck: the gate with its usage buffers and
    # diagnostics, post_norm's gain and the decoder scale g_D.  Composes with
    # tie_encoder_decoder, which then leaves a single matrix for the stack.
    share_projections: bool = False

    # LapSum barrier solve: bisection stops at this absolute residual on
    # sum p_i = K, in solver_dtype (float32 keeps the boundary reproducible
    # under bf16 training).
    barrier_solver_tol: float = 1.0e-6
    solver_dtype: str = "float32"
    log_diagnostics: bool = True

    bias: bool = False  # biasless projections (the family convention)
    # skip the whole soft-mask machinery when not training
    hard_inference: bool = True

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        from .bottleneck.controller import parse_placements

        parse_placements(self.placement)  # raises on an unknown or empty name

        from .bottleneck.module import _RENAMED_INIT_MODES, INIT_MODES

        if self.init_mode in _RENAMED_INIT_MODES:
            raise ValueError(
                f"init_mode={self.init_mode!r} was renamed to "
                f"{_RENAMED_INIT_MODES[self.init_mode]!r}"
            )
        if self.init_mode not in INIT_MODES:
            raise ValueError(
                f"unknown bottleneck init_mode: {self.init_mode!r} "
                f"({' | '.join(INIT_MODES)})"
            )
        if self.tie_encoder_decoder and self.init_mode != "unit_norm_dictionary":
            raise ValueError(
                "tie_encoder_decoder requires init_mode='unit_norm_dictionary', "
                f"got init_mode={self.init_mode!r}"
            )
        if self.tie_encoder_decoder and self.selection_mode == "gated_topk":
            raise ValueError(
                "tie_encoder_decoder is incompatible with selection_mode="
                "'gated_topk': the value and score branches are separate matrices"
            )
        if self.selection_mode not in ("topk", "abs_topk", "gated_topk"):
            raise ValueError(
                f"unknown selection_mode: {self.selection_mode} "
                "(topk | abs_topk | gated_topk)"
            )
        if self.surrogate_mode not in ("lapsum", "rblapsum", "rblapsum_sf", "hard"):
            raise ValueError(
                f"unknown surrogate_mode: {self.surrogate_mode} "
                "(lapsum | rblapsum | rblapsum_sf | hard)"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.rblapsum_boundary_floor is None:
            # concrete float in the dumped config
            self.rblapsum_boundary_floor = 0.0
        if self.rblapsum_boundary_grad_mode is None:
            # concrete string in the dumped config; the sf default follows the
            # measured hard-mode ranking (through_rank_kappa is the stable one)
            self.rblapsum_boundary_grad_mode = (
                "through_rank_kappa" if self.surrogate_mode == "rblapsum_sf"
                else "detach")
        if self.surrogate_mode in ("rblapsum", "rblapsum_sf"):
            if self.selection_mode not in ("topk", "abs_topk"):
                raise ValueError(
                    f"surrogate_mode={self.surrogate_mode!r} requires selection_mode "
                    "'topk' or 'abs_topk' (not gated_topk)"
                )
            if self.rblapsum_boundary_grad_mode not in (
                    "detach", "project", "through_rank", "through_rank_kappa"):
                raise ValueError(
                    "rblapsum_boundary_grad_mode must be detach | project | "
                    "through_rank | through_rank_kappa, "
                    f"got {self.rblapsum_boundary_grad_mode!r}"
                )
            if not 0.0 <= self.rblapsum_rho_random_perm_prob_grad <= 1.0:
                raise ValueError(
                    "rblapsum_rho_random_perm_prob_grad must be in [0, 1], "
                    f"got {self.rblapsum_rho_random_perm_prob_grad!r}")
            if (self.rblapsum_rho_random_perm_prob_grad != 0.0
                    and self.surrogate_mode != "rblapsum"):
                raise ValueError(
                    "rblapsum_rho_random_perm_prob_grad is implemented for "
                    "surrogate_mode='rblapsum' only; leave it at 0.0 under "
                    f"{self.surrogate_mode!r}")
            if self.rblapsum_sf_value_grad not in ("pool", "support"):
                raise ValueError(
                    "rblapsum_sf_value_grad must be pool | support, "
                    f"got {self.rblapsum_sf_value_grad!r}"
                )
            if (self.rblapsum_sf_value_grad != "pool"
                    and self.surrogate_mode != "rblapsum_sf"):
                raise ValueError(
                    "rblapsum_sf_value_grad is a soft-forward knob and is not "
                    f"applied by surrogate_mode={self.surrogate_mode!r}; "
                    "leave it at 'pool'"
                )
        if self.solver_dtype not in ("float32", "float64"):
            raise ValueError(f"unknown solver_dtype: {self.solver_dtype} (float32 | float64)")
        # shape rules live with the gate so the module can be built standalone
        from .bottleneck.gate import validate_gate_shapes

        validate_gate_shapes(self.n_features, self.k, self.j, self.surrogate_mode)


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    activation_bottleneck: ActivationBottleneckConfig = field(
        default_factory=ActivationBottleneckConfig
    )

    def __post_init__(self) -> None:
        # a single source of truth for the context length
        self.model.max_seq_len = max(self.model.max_seq_len, self.data.seq_len)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _read_yaml(path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Read a YAML file, resolving an optional ``_base_`` (str or list)."""
    _seen = _seen or set()
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"circular _base_ include at {path}")
    _seen.add(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        _deep_update(merged, _read_yaml(b_path, _seen))
    _deep_update(merged, raw)
    return merged


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of a YAML/CLI value to a dataclass field type."""
    origin = getattr(target_type, "__origin__", None)
    args = getattr(target_type, "__args__", ())

    if origin is None and target_type in (Any, None):
        return value

    # Optional[X] / Union[X, None]
    if origin is not None and type(None) in args:
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0]) if len(inner) == 1 else value

    if origin in (list, List):
        item_t = args[0] if args else Any
        if isinstance(value, str):
            # a bare scalar is a one-element list
            return [_coerce(value, item_t)]
        return [_coerce(v, item_t) for v in value]
    if origin in (tuple, Tuple):
        return tuple(value)
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise TypeError(f"expected a mapping, got {value!r}")
        key_t, val_t = (args + (Any, Any))[:2]
        return {str(k): _coerce(v, val_t) for k, v in value.items()}

    if target_type is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"cannot parse bool from {value!r}")
        return bool(value)
    if target_type is int:
        return int(float(value)) if isinstance(value, str) else int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _from_dict(cls: Any, data: Dict[str, Any]) -> Any:
    # `from __future__ import annotations` turns field.type into a string, so
    # resolve the real types once per dataclass.
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    for name in known:
        if name not in data:
            continue
        value = data[name]
        ftype = hints[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = _coerce(value, ftype)
    return cls(**kwargs)


def _parse_cli_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    # Shells eat quotes, so `--activation_bottleneck.layers=[0,1]` arrives as
    # `[mlp,attn]`.  Accept that, and bare `mlp,attn`, as a list of scalars.
    stripped = text.strip()
    inner = stripped[1:-1] if stripped.startswith("[") and stripped.endswith("]") else None
    if inner is None and "," in stripped:
        inner = stripped
    if inner is not None:
        items = [i.strip().strip("'\"") for i in inner.split(",") if i.strip()]
        return [_parse_cli_value(i) if i.strip("-").replace(".", "").isdigit() else i for i in items]
    return text


def apply_overrides(tree: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``a.b=value`` strings (with or without a leading ``--``)."""
    for item in overrides:
        item = item[2:] if item.startswith("--") else item
        if "=" not in item:
            raise ValueError(f"override {item!r} must have the form section.field=value")
        key, raw = item.split("=", 1)
        node = tree
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {key!r}")
        node[parts[-1]] = _parse_cli_value(raw)
    return tree


def load_config(path: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    tree: Dict[str, Any] = _read_yaml(path) if path else {}
    tree = apply_overrides(tree, overrides)
    _migrate_legacy(tree)
    return _from_dict(Config, tree)


def config_from_dict(tree: Dict[str, Any]) -> Config:
    """A Config from a *saved payload* (``config.json`` / a checkpoint).

    Same migrations as ``load_config``, plus the pins that only make sense for
    something that was already trained -- see ``_pin_legacy_logit_scale``.
    """
    tree = copy.deepcopy(tree)
    _migrate_legacy(tree)
    _pin_legacy_logit_scale(tree)
    return _from_dict(Config, tree)


def _migrate_legacy(tree: Dict[str, Any]) -> None:
    # the weight-sparsity subsystem was removed 2026-09-28 (cleanup); archived
    # configs carry a full "sparsity" block -- every archived run had
    # enabled=false, so dropping it reproduces the run exactly
    if isinstance(tree.get("sparsity"), dict):
        if tree["sparsity"].get("enabled"):
            raise ValueError(
                "this config enables the removed weight-sparsity subsystem; "
                "check out a commit before the 2026-09-28 cleanup to use it")
        del tree["sparsity"]
    ab = tree.get("activation_bottleneck")
    if isinstance(ab, dict):
        _REMOVED_AB = (
            "swap_lambda", "jumprelu_kernel_width", "jumprelu_count_coef",
            "jumprelu_theta_init", "jumprelu_count_one_sided",
            "reinforce_distribution", "reinforce_temperature",
            "reinforce_baseline", "reinforce_baseline_ema_decay",
            "reinforce_coef", "reinforce_stochastic_eval",
        )
        for k in _REMOVED_AB:
            ab.pop(k, None)
        mode = ab.get("surrogate_mode")
        if mode in ("swap_gibbs", "jumprelu", "reinforce_topk"):
            raise ValueError(
                f"surrogate_mode={mode!r} was removed in the "
                "2026-09-28 cleanup; check out an earlier commit to rebuild "
                "such a run")
        if (mode == "lapsum_scheduled"
                and ab.get("temperature_schedule") == "constant"
                and ab.get("temperature_scale_mode") == "absolute"):
            # a constant absolute schedule IS the surviving constant-T mode
            ab["surrogate_mode"] = mode = "lapsum"
            ab.setdefault("temperature", ab.get("temperature_start", 1.0))
        if mode in ("lapsum_adaptive", "lapsum_scheduled"):
            raise ValueError(
                f"surrogate_mode={mode!r} (adaptive/scheduled temperature) was "
                "removed in the 2026-09-28 cleanup; only the constant-"
                "temperature 'lapsum' survives.  Check out an earlier commit "
                "to rebuild such a run")
        if mode == "lapsum_fixed":
            if ab.get("temperature_scale_mode", "relative") != "absolute":
                raise ValueError(
                    "lapsum_fixed with a RELATIVE temperature scale was removed "
                    "in the 2026-09-28 cleanup; check out an earlier commit")
            ab["surrogate_mode"] = "lapsum"
            ab.setdefault("temperature", ab.get("fixed_temperature", 1.0))
        if ab.get("rblapsum_temperature_mode") == "servo":
            raise ValueError(
                "the rblapsum temperature servo was removed in the 2026-09-28 "
                "cleanup; check out an earlier commit to rebuild such a run")
        if "temperature" not in ab and "rblapsum_temperature" in ab:
            ab["temperature"] = ab["rblapsum_temperature"]
        if float(ab.get("reconstruction_coef", 0.0) or 0.0) != 0.0:
            raise ValueError("reconstruction_coef was removed in the 2026-09-28 "
                             "cleanup; check out an earlier commit")
        if ab.get("calibrate_output"):
            raise ValueError("calibrate_output was removed in the 2026-09-28 "
                             "cleanup; check out an earlier commit")
        for k in ("n_eff", "effective_count_metric", "boundary_mode",
                  "one_sided_weight_mode", "fixed_temperature",
                  "temperature_schedule", "temperature_start", "temperature_end",
                  "temperature_warmup_steps", "temperature_anneal_steps",
                  "temperature_power", "temperature_scale_mode",
                  "temperature_solver_tol", "temperature_solver_max_iters",
                  "differentiate_temperature", "project_scale_gradient",
                  "inactive_grad_scale", "surrogate_grad_scale",
                  "rblapsum_temperature", "rblapsum_kernel",
                  "rblapsum_temperature_mode", "rblapsum_chi_target",
                  "rblapsum_window_floor", "rblapsum_t_min", "rblapsum_t_max",
                  "rblapsum_servo_rate", "reconstruction_coef",
                  "reconstruction_normalize", "calibrate_output",
                  "calibration_batches", "calibration_iters"):
            ab.pop(k, None)


def _pin_legacy_logit_scale(tree: Dict[str, Any]) -> None:
    """Pin pre-``logit_scale`` checkpoints to the behaviour they were trained with.

    A checkpoint saved before ``logit_scale`` existed was trained with a tied head
    whose logits were *not* rescaled, so letting it pick up the "auto" default
    would shrink its logits by ``init_std / init_std_embedding`` (~35x) and turn
    sampled generations to noise.  Absence of the key dates the payload.

    Saved payloads only (``config_from_dict``): a hand-written YAML that simply
    does not mention the field must get the dataclass default instead, or every
    shipped config silently trains an unscaled tied head.
    """
    model = tree.get("model")
    if isinstance(model, dict) and "logit_scale" not in model:
        model["logit_scale"] = "none"
