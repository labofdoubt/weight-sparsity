"""Orthogonal / tight-frame bottleneck initialization on the MD path.

Covers the geometry itself (Gram isotropy, unit mean row/column norms, the
Frobenius norm the MD sphere expects), the fixed global decoder scale g_D, and
the one thing that could silently go wrong: whether g_D reaches the MD
parameter gradients with the right factor.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.bottleneck.module import effective_backward_support
from wsparse.config import ActivationBottleneckConfig, Config, ModelConfig
from wsparse.decouple import (
    RAW_GAIN_ONE,
    DecoupledAdamW,
    build_decoupled_optimizer,
    md_init_,
)
from wsparse.model import build_model

D_MODEL, N_FEATURES = 32, 128


class _Train:
    lr = 3e-3
    betas = (0.9, 0.95)
    eps = 1e-8


def build(bottleneck_init="orthogonal", decoder_scale="none", surrogate="hard",
          k=8, j=8, n_features=N_FEATURES, d_model=D_MODEL, seed=0, **bn_kw):
    """A tiny MD model with one bottleneck per block, initialized."""
    model_cfg = ModelConfig(
        vocab_size=97, max_seq_len=32, n_layers=2, d_model=d_model, n_heads=4,
        pos_encoding="rope", decouple=True, logit_scale="none", bias=False,
        bottleneck_init=bottleneck_init, bottleneck_decoder_scale=decoder_scale)
    fields = dict(enabled=True, layers="all", placement="residual_out",
                  n_features=n_features, k=k, j=j, surrogate_mode=surrogate,
                  temperature=1.0, bias=False)
    fields.update(bn_kw)  # callers may override any of the above
    bn_cfg = ActivationBottleneckConfig(**fields)
    torch.manual_seed(seed)
    model = build_model(model_cfg)
    ctl = apply_activation_bottleneck(model, bn_cfg, max_steps=10)
    stats = md_init_(model, model_cfg.decouple_gains)
    return model, ctl, stats


def gram_rel_err(W, d_model, diag):
    """||W W^T - diag I||_F / ||diag I||_F, on whichever side is square."""
    W = W.detach().float()
    gram = W.t() @ W if W.shape[1] == d_model else W @ W.t()
    target = diag * torch.eye(d_model)
    return float((gram - target).norm() / target.norm())


# --------------------------------------------------------------------------- #
# the MD convention (section 8 of the spec): c_F must already be sqrt(d_b)
# --------------------------------------------------------------------------- #

def test_md_sphere_radius_is_sqrt_d_bottleneck():
    """Both bottleneck matrices have c_F = sqrt(d_b), which a tight frame meets.

    c_F = sqrt(d_out * d_in / d_model) and each matrix has one axis equal to
    d_model, so c_F = sqrt(d_b) for the encoder AND the decoder -- exactly the
    Frobenius norm of a (d_b/d)-scaled frame, tr((d_b/d) I_d) = d_b.  The
    orthogonal init therefore needs no change to the MD parameterization.
    """
    model, ctl, _ = build()
    mod = ctl.layers[0][1]
    for W in (mod.in_proj.weight, mod.out_proj.weight):
        c_f = math.sqrt(W.shape[0] * W.shape[1] / D_MODEL)
        assert c_f == pytest.approx(math.sqrt(N_FEATURES))
        assert float(W.detach().norm()) == pytest.approx(c_f, rel=1e-5)


def test_optimizer_captures_that_same_radius():
    """The sphere the optimizer captures is unchanged by the new init."""
    model, ctl, _ = build()
    opt = build_decoupled_optimizer(model, _Train())
    x = torch.randint(0, 97, (2, 8))
    for g in opt.param_groups:
        g["lr"] = 0.0
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()  # materializes state without moving anything
    mod = ctl.layers[0][1]
    for W in (mod.in_proj.weight, mod.out_proj.weight):
        assert float(opt.state[W]["c_f"]) == pytest.approx(
            math.sqrt(N_FEATURES), rel=1e-5)


# --------------------------------------------------------------------------- #
# the geometry
# --------------------------------------------------------------------------- #

def test_encoder_is_a_tight_frame():
    model, ctl, stats = build()
    W = ctl.layers[0][1].in_proj.weight.detach()
    assert W.shape == (N_FEATURES, D_MODEL)
    assert gram_rel_err(W, D_MODEL, N_FEATURES / D_MODEL) < 1e-5
    assert float((W ** 2).sum(dim=1).mean()) == pytest.approx(1.0, rel=1e-5)
    sv = torch.linalg.svdvals(W.float())
    assert float(sv.max() / sv.min()) == pytest.approx(1.0, rel=1e-4)
    assert stats["encoder"]["gram_rel_err"] < 1e-5


def test_decoder_is_a_tight_frame():
    model, ctl, stats = build()
    W = ctl.layers[0][1].out_proj.weight.detach()
    assert W.shape == (D_MODEL, N_FEATURES)
    assert gram_rel_err(W, D_MODEL, N_FEATURES / D_MODEL) < 1e-5
    assert float((W ** 2).sum(dim=0).mean()) == pytest.approx(1.0, rel=1e-5)
    assert stats["decoder"]["gram_rel_err"] < 1e-5


def test_encoder_and_decoder_are_independent():
    """Not tied and not transposes of one another."""
    model, ctl, _ = build()
    mod = ctl.layers[0][1]
    enc, dec = mod.in_proj.weight.detach(), mod.out_proj.weight.detach()
    # if they were the same frame, enc @ dec would be ~(d_b/d) I
    prod = dec @ enc  # (d, d)
    assert gram_rel_err(prod, D_MODEL, N_FEATURES / D_MODEL) > 0.1
    assert not torch.allclose(dec, enc.t(), atol=1e-3)


def test_layers_get_different_frames():
    model, ctl, _ = build()
    a = ctl.layers[0][1].in_proj.weight.detach()
    b = ctl.layers[1][1].in_proj.weight.detach()
    assert not torch.allclose(a, b, atol=1e-3)


def test_standard_init_keeps_its_random_directions():
    """The default path is untouched: unit mean norms, but NOT a tight frame."""
    model, ctl, stats = build(bottleneck_init="standard")
    mod = ctl.layers[0][1]
    enc, dec = mod.in_proj.weight.detach(), mod.out_proj.weight.detach()
    assert float((enc ** 2).sum(dim=1).mean()) == pytest.approx(1.0, rel=1e-3)
    assert float((dec ** 2).sum(dim=0).mean()) == pytest.approx(1.0, rel=1e-3)
    assert gram_rel_err(enc, D_MODEL, N_FEATURES / D_MODEL) > 0.1
    assert gram_rel_err(dec, D_MODEL, N_FEATURES / D_MODEL) > 0.1
    assert mod.decoder_scale == 1.0
    assert "encoder" not in stats  # frame stats are for the orthogonal mode
    # and it is deterministic at a fixed seed
    _, ctl2, _ = build(bottleneck_init="standard")
    assert torch.equal(enc, ctl2.layers[0][1].in_proj.weight.detach())


# --------------------------------------------------------------------------- #
# K_eff and g_D
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("surrogate,k,j,want", [
    ("hard", 8, 8, 8), ("hard", 8, 0, 8),
    ("lapsum", 8, 8, 16), ("rblapsum", 8, 24, 32), ("rblapsum_sf", 4, 12, 16),
])
def test_effective_backward_support(surrogate, k, j, want):
    cfg = ActivationBottleneckConfig(enabled=True, n_features=64, k=k, j=j,
                                     surrogate_mode=surrogate, temperature=1.0)
    assert effective_backward_support(cfg) == float(want)


def test_decoder_scale_none_is_one():
    model, ctl, stats = build(decoder_scale="none")
    assert stats["g_D"] == 1.0
    assert all(m.decoder_scale == 1.0 for _, m in ctl.layers)


@pytest.mark.parametrize("surrogate,k,j", [("hard", 8, 8), ("rblapsum", 8, 24)])
def test_backward_preserving_scale(surrogate, k, j):
    k_eff = k if surrogate == "hard" else k + j
    model, ctl, stats = build(decoder_scale="backward_preserving",
                              surrogate=surrogate, k=k, j=j)
    want = math.sqrt(D_MODEL / k_eff)
    assert stats["k_eff"] == float(k_eff)
    assert stats["g_D"] == pytest.approx(want)
    for _, mod in ctl.layers:
        assert mod.decoder_scale == pytest.approx(want)
        # the EFFECTIVE decoder is a frame with diagonal d_b / K_eff
        W_eff = mod.out_proj.weight.detach() * mod.decoder_scale
        assert gram_rel_err(W_eff, D_MODEL, N_FEATURES / k_eff) < 1e-5
        assert float((W_eff ** 2).sum(dim=0).mean()) == pytest.approx(
            D_MODEL / k_eff, rel=1e-4)
    # and it does not leak into the gains or the sphere
    opt = build_decoupled_optimizer(model, _Train())
    for g in opt.param_groups:
        g["lr"] = 0.0
    x = torch.randint(0, 97, (2, 8))
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    st = opt.state[ctl.layers[0][1].out_proj.weight]
    assert torch.all(st["raw_grow"] == RAW_GAIN_ONE)
    assert torch.all(st["raw_gcol"] == RAW_GAIN_ONE)
    assert float(st["c_f"]) == pytest.approx(math.sqrt(N_FEATURES), rel=1e-5)


def test_decode_scales_the_weight_not_the_bias():
    """g_D multiplies W_D; the decoder bias keeps its own scale."""
    model, ctl, _ = build(decoder_scale="backward_preserving", bias=True)
    mod = ctl.layers[0][1]
    torch.manual_seed(1)
    with torch.no_grad():
        mod.out_proj.bias.normal_()
    code = torch.randn(4, N_FEATURES)
    want = F.linear(code, mod.out_proj.weight * mod.decoder_scale,
                    mod.out_proj.bias)
    assert torch.allclose(mod.decode(code), want, atol=1e-5)


# --------------------------------------------------------------------------- #
# g_D in the MD gradients
# --------------------------------------------------------------------------- #

class _CapturingAdamW(DecoupledAdamW):
    """Records every (prefix, gradient) the MD step feeds to Adam."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captured = {}

    def _adam_(self, value, grad, state, prefix, *a, **kw):
        self.captured[prefix] = grad.detach().clone()
        return super()._adam_(value, grad, state, prefix, *a, **kw)


@pytest.mark.parametrize("g_D", [1.0, 2.5])
def test_md_gradient_split_matches_autograd_with_gD(g_D):
    """The manual MD chain rule equals autograd, g_D included.

    The model stores the fused P = diag(g_row) W_hat diag(g_col) and the
    optimizer splits dL/dP by hand, so the only way g_D can reach W_hat and the
    gains is through the forward graph putting it into dL/dP.  This checks the
    whole path against autograd on an explicit parameterization
    W_eff = g_D * diag(g_row) W_hat diag(g_col).
    """
    torch.manual_seed(0)
    d_out, d_in = 5, 7
    w_hat = torch.randn(d_out, d_in, dtype=torch.float64)
    raw_r = torch.randn(d_out, dtype=torch.float64) * 0.3 + RAW_GAIN_ONE
    raw_c = torch.randn(d_in, dtype=torch.float64) * 0.3 + RAW_GAIN_ONE
    A = torch.randn(d_out, d_in, dtype=torch.float64)  # dL/dW_eff for L = <W_eff, A>

    # --- autograd reference on the explicit parameterization --------------- #
    wh = w_hat.clone().requires_grad_(True)
    rr = raw_r.clone().requires_grad_(True)
    rc = raw_c.clone().requires_grad_(True)
    W_eff = g_D * (F.softplus(rr).unsqueeze(1) * wh * F.softplus(rc).unsqueeze(0))
    (W_eff * A).sum().backward()

    # --- what the optimizer computes from the fused weight ----------------- #
    fused = (F.softplus(raw_r).unsqueeze(1) * w_hat
             * F.softplus(raw_c).unsqueeze(0))
    p = torch.nn.Parameter(fused.clone())
    p.grad = g_D * A.clone()  # what a forward carrying g_D delivers for dL/dP
    opt = _CapturingAdamW(
        [dict(params=[p], kind="md", lr=0.0, name="md", weight_decay=0.0,
              is_mask=False)],
        betas=(0.9, 0.95), eps=1e-8, gain_mode="row_col")
    st = opt._state_for(p, "md", "row_col")
    st["raw_grow"].copy_(raw_r)
    st["raw_gcol"].copy_(raw_c)
    opt.step()

    assert torch.allclose(opt.captured[""], wh.grad, atol=1e-10), "direction"
    assert torch.allclose(opt.captured["grow_"], rr.grad, atol=1e-10), "row gain"
    assert torch.allclose(opt.captured["gcol_"], rc.grad, atol=1e-10), "col gain"


def test_forward_graph_puts_gD_into_the_stored_weights_gradient():
    """dL/dP carries exactly one factor of g_D, at a fixed upstream gradient.

    Measured on the module alone with a fixed cotangent: g_D changes the whole
    network's forward, so comparing full-model gradients would confound the
    factor with a different dL/dy.  The encoder's gradient scales too, since it
    reaches the loss only through the decoder.
    """
    model, ctl, _ = build(decoder_scale="backward_preserving", k=8, j=8)
    mod = ctl.layers[0][1]
    g_D = mod.decoder_scale
    assert g_D != 1.0
    torch.manual_seed(0)
    x = torch.randn(4, 6, D_MODEL)
    cotangent = torch.randn(4, 6, D_MODEL)

    grads = {}
    for scale in (g_D, 1.0):
        mod.decoder_scale = scale
        mod.zero_grad(set_to_none=True)
        mod(x).backward(cotangent)
        grads[scale] = (mod.out_proj.weight.grad.detach().clone(),
                        mod.in_proj.weight.grad.detach().clone())
    mod.decoder_scale = g_D
    for which, (a, b) in enumerate(zip(grads[g_D], grads[1.0])):
        assert float(a.norm() / b.norm()) == pytest.approx(g_D, rel=1e-4), which
    # and the same support was selected either way (g_D never touches scores)
    assert torch.allclose(grads[g_D][1], g_D * grads[1.0][1], atol=1e-5)


# --------------------------------------------------------------------------- #
# the three configurations run
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bn_init,scale", [
    ("standard", "none"), ("orthogonal", "none"),
    ("orthogonal", "backward_preserving"),
])
@pytest.mark.parametrize("surrogate", ["hard", "rblapsum"])
def test_two_steps_of_md_training(bn_init, scale, surrogate):
    model, ctl, _ = build(bottleneck_init=bn_init, decoder_scale=scale,
                          surrogate=surrogate, k=8, j=8)
    opt = build_decoupled_optimizer(model, _Train())
    x = torch.randint(0, 97, (2, 8))
    losses = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(math.isfinite(v) for v in losses)
    # the direction stayed on its sphere; the scale did not drift into it
    for _, mod in ctl.layers:
        for W in (mod.in_proj.weight, mod.out_proj.weight):
            st = opt.state[W]
            w_hat = W.detach() / (F.softplus(st["raw_grow"]).unsqueeze(1)
                                  * F.softplus(st["raw_gcol"]).unsqueeze(0))
            assert float(w_hat.norm()) == pytest.approx(float(st["c_f"]), rel=1e-4)


def test_orthogonal_rejects_a_tied_decoder():
    with pytest.raises(ValueError, match="tied decoder"):
        build(n_features=D_MODEL, tie_encoder_decoder=True,
              init_mode="unit_norm_dictionary")


def test_orthogonal_requires_the_md_path():
    with pytest.raises(ValueError, match="MD path"):
        ModelConfig(bottleneck_init="orthogonal")
    with pytest.raises(ValueError, match="MD initialization"):
        ModelConfig(bottleneck_decoder_scale="backward_preserving")


def test_config_roundtrip_and_defaults():
    cfg = Config()
    assert cfg.model.bottleneck_init == "standard"
    assert cfg.model.bottleneck_decoder_scale == "none"
    tree = cfg.to_dict()
    assert tree["model"]["bottleneck_init"] == "standard"
