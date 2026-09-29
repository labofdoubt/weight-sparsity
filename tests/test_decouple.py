"""Magnitude-direction decoupling: init invariants, the optimizer step, resume."""

import math

import pytest
import torch
import torch.nn.functional as F

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import ActivationBottleneckConfig, ModelConfig, config_from_dict
from wsparse.decouple import (
    RAW_GAIN_ONE,
    DecoupledAdamW,
    build_decoupled_optimizer,
    md_init_,
    md_spread_gain_,
)
from wsparse.model import build_model


def tiny_cfg(**kw):
    base = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4,
                mlp_ratio=4.0, pos_encoding="rope", decouple=True, logit_scale="none")
    base.update(kw)
    return ModelConfig(**base)


class _Train:
    lr = 3e-3
    betas = (0.9, 0.95)
    eps = 1e-8


def _model(gains="row_col", bottleneck=False):
    torch.manual_seed(0)
    cfg = tiny_cfg(decouple_gains=gains)
    model = build_model(cfg)
    if bottleneck:
        bn = ActivationBottleneckConfig(
            enabled=True, layers="all", placement="residual_out", n_features=64,
            k=4, j=4, surrogate_mode="lapsum", temperature=1.0, bias=False,
            # the dc runs' calibration pairing; the default one-sided mode
            # additionally demands 1 < n_eff < j, which k=j=4 cannot satisfy,
        )
        apply_activation_bottleneck(model, bn, max_steps=10)
    md_init_(model, gains)
    return model


def _check_constraints(model, opt=None, tol=1e-3):
    """Embeddings on unit rows; every matrix direction on its c_F sphere."""
    d = model.cfg.d_model
    embed_ids = {id(model.tok_emb.weight), id(model.lm_head.weight)}
    rows = model.tok_emb.weight.norm(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    seen = set()
    for p in model.parameters():
        if id(p) in seen or id(p) in embed_ids or p.dim() < 2:
            continue
        seen.add(id(p))
        c_f = math.sqrt(p.shape[0] * p.shape[1] / d)
        w_hat = p.detach().clone()
        if opt is not None and p in opt.state and opt.state[p]:
            st = opt.state[p]
            if "raw_grow" in st:
                w_hat = w_hat / F.softplus(st["raw_grow"]).unsqueeze(1)
            if "raw_gcol" in st:
                w_hat = w_hat / F.softplus(st["raw_gcol"]).unsqueeze(0)
        assert abs(w_hat.norm().item() - c_f) < tol * c_f, (p.shape, w_hat.norm().item(), c_f)


def test_md_init_overrides_everything_including_the_bottleneck():
    model = _model(bottleneck=True)
    _check_constraints(model)
    d = model.cfg.d_model
    # entrywise std is 1/sqrt(d_model) regardless of fan-in -- unlike both the
    # model's fan_in scheme and the bottleneck's selection-corrected init
    fc2 = model.blocks[0].mlp.fc2.weight       # fan-in d_mlp != d_model
    assert abs(fc2.std().item() - 1 / math.sqrt(d)) < 0.15 / math.sqrt(d)
    bn = model.blocks[0].residual_out_bottleneck
    for W in (bn.in_proj.weight, bn.out_proj.weight):
        c_f = math.sqrt(W.shape[0] * W.shape[1] / d)
        assert abs(W.norm().item() - c_f) < 1e-4 * c_f
    # the sqrt(d) input upscale is on
    assert abs(model.embed_scale - math.sqrt(d)) < 1e-9


def test_step_preserves_constraints_and_learns():
    torch.manual_seed(0)
    model = _model(bottleneck=True)
    opt = build_decoupled_optimizer(model, _Train())
    x = torch.randint(0, 97, (4, 16))
    losses = []
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        _check_constraints(model, opt)
    assert losses[-1] < losses[0] - 0.5, losses[::6]
    # gains actually moved off their init of exactly 1
    moved = [
        (F.softplus(st["raw_grow"]) - 1).abs().max().item()
        for st in opt.state.values() if "raw_grow" in st
    ]
    assert moved and max(moved) > 1e-3


@pytest.mark.parametrize("gains,fc1,fc2,proj", [
    ("row_col", ("raw_grow", "raw_gcol"), ("raw_grow", "raw_gcol"), ("raw_grow", "raw_gcol")),
    # up_down: d_out >= d_in -> row gain; d_out < d_in -> column gain.  fc1 is
    # an up-projection, fc2 a down-projection, attn.proj square (counts as up).
    ("up_down", ("raw_grow",), ("raw_gcol",), ("raw_grow",)),
])
def test_gain_placement(gains, fc1, fc2, proj):
    torch.manual_seed(0)
    model = _model(gains=gains)
    opt = build_decoupled_optimizer(model, _Train(), gain_mode=gains)
    x = torch.randint(0, 97, (2, 8))
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    blk = model.blocks[0]
    for mod, expect in ((blk.mlp.fc1, fc1), (blk.mlp.fc2, fc2), (blk.attn.proj, proj)):
        st = opt.state[mod.weight]
        have = tuple(k for k in ("raw_grow", "raw_gcol") if k in st)
        assert have == expect, (tuple(mod.weight.shape), have, expect)


def _materialize_state(model, opt, lr=0.0):
    """Create optimizer state without moving anything (state is lazy).

    A zero-lr step is also the sharpest check of the decomposition: the step
    recovers W_hat from the fused weight, projects it onto c_F and re-fuses, so
    if c_F did not match the direction's norm the fused weight would come back
    rescaled.
    """
    saved = [g["lr"] for g in opt.param_groups]
    for g in opt.param_groups:
        g["lr"] = lr
    x = torch.randint(0, 97, (2, 8))
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    for g, old in zip(opt.param_groups, saved):
        g["lr"] = old
    opt.zero_grad(set_to_none=True)


@pytest.mark.parametrize("gains,per_matrix", [("row_col", 4.0), ("up_down", 9.0)])
def test_spread_gain_lands_in_the_gains_not_the_sphere(gains, per_matrix):
    """md_spread_gain_ must scale the fused weight and start the gains there.

    The factor has to end up in softplus(raw), with c_F still the direction's
    own norm, or the decomposition W = diag(g_row) W_hat diag(g_col) no longer
    holds at step 0 and the sphere silently freezes the scale.
    """
    torch.manual_seed(0)
    model = _model(gains=gains, bottleneck=True)
    mods = [m for m in model.modules() if hasattr(m, "in_proj") and hasattr(m, "gate")]
    assert mods, "no bottleneck modules found"
    n_gains = 2 if gains == "row_col" else 1
    before = {}
    for m in mods:
        for proj in (m.in_proj, m.out_proj):
            before[id(proj.weight)] = proj.weight.detach().clone()
            g0 = md_spread_gain_(proj.weight, per_matrix, gains)
            want = per_matrix ** (1.0 / n_gains)
            assert [g for g in g0 if g != 1.0] == pytest.approx([want] * n_gains)

    opt = build_decoupled_optimizer(model, _Train(), gain_mode=gains)
    _materialize_state(model, opt)

    for m in mods:
        for proj in (m.in_proj, m.out_proj):
            w = proj.weight
            # the fused weight moved by exactly the requested factor, and the
            # zero-lr step did not move it back
            assert torch.allclose(w, before[id(w)] * per_matrix, atol=1e-5)
            st = opt.state[w]
            gain_keys = [k for k in ("raw_grow", "raw_gcol") if k in st]
            assert len(gain_keys) == n_gains
            want = per_matrix ** (1.0 / n_gains)
            for k in gain_keys:
                g = F.softplus(st[k])
                assert torch.allclose(g, torch.full_like(g, want), atol=1e-5)
            # c_F is the direction's norm, not the fused one
            w_hat = w.detach().float()
            if "raw_grow" in st:
                w_hat = w_hat / F.softplus(st["raw_grow"]).unsqueeze(1)
            if "raw_gcol" in st:
                w_hat = w_hat / F.softplus(st["raw_gcol"]).unsqueeze(0)
            assert float(w_hat.norm()) == pytest.approx(float(st["c_f"]), rel=1e-4)

    # and it still trains: one real step keeps every constraint
    x = torch.randint(0, 97, (2, 8))
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    _check_constraints(model, opt)


@pytest.mark.parametrize("where,axis", [("col", "raw_gcol"), ("row", "raw_grow")])
def test_gain_placement_in_one_vector(where, axis):
    """All of the factor in one gain vector, the other left at 1.

    The fused weight is the same as a split placement, so nothing at step 0
    depends on this; what changes is which parameters carry the scale.
    """
    torch.manual_seed(0)
    model = _model(bottleneck=True)
    mods = [m for m in model.modules() if hasattr(m, "in_proj") and hasattr(m, "gate")]
    scale = 9.0
    for m in mods:
        md_spread_gain_(m.out_proj.weight, scale, "row_col", where=where)
    opt = build_decoupled_optimizer(model, _Train(), gain_mode="row_col")
    _materialize_state(model, opt)
    other = "raw_grow" if axis == "raw_gcol" else "raw_gcol"
    for m in mods:
        st = opt.state[m.out_proj.weight]
        g = F.softplus(st[axis])
        assert torch.allclose(g, torch.full_like(g, scale), atol=1e-4)
        g_other = F.softplus(st[other])
        assert torch.allclose(g_other, torch.ones_like(g_other), atol=1e-6)
        # the encoder was not touched at all
        st_enc = opt.state[m.in_proj.weight]
        for k in ("raw_grow", "raw_gcol"):
            assert torch.all(st_enc[k] == RAW_GAIN_ONE)
    _check_constraints(model, opt)


def test_untagged_matrices_start_at_gain_one():
    """The default path is untouched: raw gains RAW_GAIN_ONE, c_F = ||W||."""
    torch.manual_seed(0)
    model = _model()
    w = model.blocks[0].mlp.fc1.weight
    norm0 = float(w.detach().float().norm())
    assert not hasattr(w, "_md_gain0")
    opt = build_decoupled_optimizer(model, _Train())
    _materialize_state(model, opt)
    st = opt.state[w]
    for k in ("raw_grow", "raw_gcol"):
        assert torch.all(st[k] == RAW_GAIN_ONE)
    assert float(st["c_f"]) == pytest.approx(norm0, rel=1e-6)


def test_resume_roundtrip():
    torch.manual_seed(0)
    model = _model()
    opt = build_decoupled_optimizer(model, _Train())
    x = torch.randint(0, 97, (2, 8))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        opt.step()
    payload = opt.state_dict()
    opt2 = build_decoupled_optimizer(model, _Train())
    opt2.load_state_dict(payload)
    # c_F must survive the roundtrip -- it defines the sphere
    p = model.blocks[0].mlp.fc1.weight
    assert torch.allclose(opt2.state[p]["c_f"], opt.state[p]["c_f"])
    opt2.zero_grad(set_to_none=True)
    _, loss = model(x, x)
    loss.backward()
    opt2.step()
    _check_constraints(model, opt2)


def test_config_validation():
    with pytest.raises(ValueError):
        tiny_cfg(pos_encoding="learned")            # decouple needs rope
    with pytest.raises(ValueError):
        tiny_cfg(logit_scale="auto")                # auto is init-derived
    with pytest.raises(ValueError):
        tiny_cfg(decouple_gains="diagonal")
    # old configs default off
    cfg = config_from_dict({"model": {"vocab_size": 97, "max_seq_len": 32,
                                      "n_layers": 1, "d_model": 32, "n_heads": 4}})
    assert cfg.model.decouple is False


def test_softplus_raw_gain_one():
    assert abs(F.softplus(torch.tensor(RAW_GAIN_ONE)).item() - 1.0) < 1e-7
