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
            k=4, j=4, n_eff=4.0, surrogate_mode="lapsum_scheduled", bias=False,
            # the dc runs' calibration pairing; the default one-sided mode
            # additionally demands 1 < n_eff < j, which k=j=4 cannot satisfy
            boundary_mode="both_sides", one_sided_weight_mode="true_gradient",
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
