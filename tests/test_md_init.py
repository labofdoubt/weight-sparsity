"""md_init: the MD initialization under a plain optimizer.

The contract is exact: at the same seed, a ``md_init=True`` model starts from
bitwise the same tensors as a ``decouple=True`` model (same ``md_init_`` call at
the same point of the same construction sequence) and computes the same forward
(same ``sqrt(d)`` embedding upscale) -- but it then trains as ordinary AdamW:
no gains, no re-projection, weight decay as configured.
"""

import math

import pytest
import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.decouple import build_decoupled_optimizer, md_init_
from wsparse.model import build_model
from wsparse.optim import build_optimizer


def tiny_cfg(**kw):
    base = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4,
                mlp_ratio=4.0, pos_encoding="rope", logit_scale="none")
    base.update(kw)
    return ModelConfig(**base)


def _bottleneck_cfg():
    return ActivationBottleneckConfig(
        enabled=True, layers="all", placement="residual_out", n_features=64,
        k=4, j=4, n_eff=4.0, surrogate_mode="lapsum_scheduled", bias=False,
        boundary_mode="both_sides", one_sided_weight_mode="true_gradient",
    )


def _build(seed, **model_kw):
    """The train()-path construction order: seed, model, bottleneck, md_init_."""
    torch.manual_seed(seed)
    cfg = tiny_cfg(**model_kw)
    model = build_model(cfg)
    apply_activation_bottleneck(model, _bottleneck_cfg(), max_steps=10)
    md_init_(model, cfg.decouple_gains)
    return model


class _Train:
    optimizer = "adamw"
    lr = 3e-3
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.1


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #


def test_md_init_excludes_decouple():
    with pytest.raises(ValueError, match="redundant"):
        tiny_cfg(md_init=True, decouple=True)


def test_md_init_requires_rope_and_no_logit_auto():
    with pytest.raises(ValueError, match="rope"):
        tiny_cfg(md_init=True, pos_encoding="learned")
    with pytest.raises(ValueError, match="logit_scale"):
        tiny_cfg(md_init=True, logit_scale="auto")


# --------------------------------------------------------------------------- #
# the init is the MD init, bit for bit
# --------------------------------------------------------------------------- #


def test_same_seed_gives_bitwise_identical_parameters():
    a = _build(0, decouple=True)
    b = _build(0, md_init=True)
    pa = dict(a.named_parameters())
    pb = dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for name in pa:
        assert torch.equal(pa[name], pb[name]), name


def test_same_seed_gives_identical_forward():
    a = _build(0, decouple=True)
    b = _build(0, md_init=True)
    assert a.embed_scale == b.embed_scale == math.sqrt(a.cfg.d_model)
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.equal(la, lb)


def test_md_init_alone_leaves_embed_scale_at_one():
    torch.manual_seed(0)
    assert build_model(tiny_cfg()).embed_scale == 1.0


# --------------------------------------------------------------------------- #
# ...but the training is plain AdamW: no projection, decay applies
# --------------------------------------------------------------------------- #


def _one_step(model, optimizer):
    x = torch.randint(0, 97, (2, 16))
    y = torch.randint(0, 97, (2, 16))
    _, ce = model(x, y)
    ce.backward()
    optimizer.step()


def test_md_init_steps_leave_the_spheres():
    model = _build(0, md_init=True)
    opt = build_optimizer(model, _Train())
    # every >=2-D weight (embeddings included) is in the decay group
    wd_by_group = {g["name"]: g["weight_decay"] for g in opt.param_groups}
    assert wd_by_group["decay"] == pytest.approx(0.1)
    _one_step(model, opt)
    rows = model.tok_emb.weight.norm(dim=-1)
    assert not torch.allclose(rows, torch.ones_like(rows), atol=1e-4)
    w = model.blocks[0].mlp.fc1.weight
    c_f = math.sqrt(w.shape[0] * w.shape[1] / model.cfg.d_model)
    assert abs(w.norm().item() - c_f) > 1e-3
    # no MD machinery in the optimizer state
    for st in opt.state.values():
        assert "raw_grow" not in st and "raw_gcol" not in st


def test_decoupled_twin_stays_on_the_spheres():
    model = _build(0, decouple=True)
    opt = build_decoupled_optimizer(model, _Train())
    _one_step(model, opt)
    rows = model.tok_emb.weight.norm(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
