"""laplace_policy with policy_estimator="rao_blackwell" (RaoBlackwellSelect).

The estimator is the conditional expectation of the likelihood-ratio sample
over each pool member's own noise, with the single-swap loss difference taken
to first order.  For a loss LINEAR in the gate output that first-order
difference is exact, so the two estimators must agree in expectation; the
tests check that by Monte Carlo (tolerance from the measured standard errors),
then the closed-form backward on one fixed draw, the value path, the
first-order scope on a two-gate chain, the scale-free width, the config rules
and a training run.
"""
import math

import numpy as np
import pytest
import torch

from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.laplace_policy import (PolicyCollector, PolicyForwardSettings,
                                               make_generator, policy_forward)
from wsparse.bottleneck.rblapsum import first_order_backward
from wsparse.config import ActivationBottleneckConfig, Config, ModelConfig
from wsparse.train import train

CPU = torch.device("cpu")


def gate_of(**kw):
    cfg = dict(n_features=16, k=3, j=5, surrogate_mode="laplace_policy", temperature=0.4,
               selection_mode="abs_topk", policy_estimator="rao_blackwell")
    cfg.update(kw)
    g = AdaptiveLapSumTopKGate(**cfg)
    g.train()
    return g


def forward(gate, z, seed, keep=False):
    col = PolicyCollector(keep_samples=keep)
    with policy_forward([gate], PolicyForwardSettings(sample=True, collector=col,
                                                       generator=make_generator(CPU, seed))):
        y = gate(z)
    return y, col


def mc_gradient(estimator, z0, w, rows, seed, **kw):
    """Mean and standard error over ``rows`` replicated rows of d(w.y)/dz."""
    gate = gate_of(policy_estimator=estimator, **kw)
    z = z0.repeat(rows, 1).clone().requires_grad_(True)
    y, col = forward(gate, z, seed)
    loss = (y * w).sum(-1)
    lp = col.records[0].log_prob
    b = loss.detach().mean()
    (g,) = torch.autograd.grad(loss.sum() + ((loss.detach() - b) * lp).sum(), [z])
    return g.mean(0), g.std(0) / math.sqrt(rows)


@pytest.mark.parametrize("width,wg,tau", [("absolute", "frozen", 0.4),
                                          ("relative_span", "through", 0.5),
                                          ("relative_b", "through", 0.3)])
def test_rao_blackwell_matches_likelihood_ratio_in_expectation_for_a_linear_loss(width, wg, tau):
    torch.manual_seed(0)
    z0 = torch.randn(16, dtype=torch.float64)
    w = torch.randn(16, dtype=torch.float64)
    kw = dict(policy_temperature_mode=width, policy_width_gradient=wg, temperature=tau)
    g_rb, se_rb = mc_gradient("rao_blackwell", z0, w, 40_000, 1, **kw)
    g_lr, se_lr = mc_gradient("likelihood_ratio", z0, w, 400_000, 2, **kw)
    se = torch.sqrt(se_rb ** 2 + se_lr ** 2)
    pool = se > 0
    assert pool.sum() == 8                      # the K+J pool members carry gradient
    zscore = ((g_rb - g_lr)[pool] / se[pool]).abs()
    assert float(zscore.max()) < 4.5, zscore
    # and the conditional estimator is the lower-variance one, per sample
    assert float((se_rb[pool] * math.sqrt(40_000)).median()) < float(
        (se_lr[pool] * math.sqrt(400_000)).median())


def test_averaging_over_further_pool_draws_keeps_the_mean_and_lowers_the_variance():
    torch.manual_seed(0)
    z0 = torch.randn(16, dtype=torch.float64)
    w = torch.randn(16, dtype=torch.float64)
    g1, se1 = mc_gradient("rao_blackwell", z0, w, 40_000, 1)
    g8, se8 = mc_gradient("rao_blackwell", z0, w, 40_000, 3, policy_rb_samples=8)
    gl, sel = mc_gradient("likelihood_ratio", z0, w, 400_000, 2)
    pool = se1 > 0
    z = ((g8 - gl)[pool] / torch.sqrt(se8 ** 2 + sel ** 2)[pool]).abs()
    assert float(z.max()) < 4.5, z
    assert float((se8[pool] / se1[pool]).median()) < 0.8


def test_backward_is_the_crossing_density_times_the_single_swap_gain():
    torch.manual_seed(3)
    gate = gate_of(policy_center_scores=False)
    z = torch.randn(1, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, 16, dtype=torch.float64)
    y, col = forward(gate, z, seed=5, keep=True)
    (g,) = torch.autograd.grad((y * w).sum(), [z])
    rec = col.records[0]
    r, idx = rec.r[0], rec.cand_idx[0]
    s = z.detach().abs()[0]
    u = s[idx]
    m = torch.zeros_like(r).scatter(-1, torch.topk(r, 3).indices, 1.0)
    v = z.detach()[0][idx]
    gw = w[0][idx]
    order = torch.argsort(r, descending=True)
    rk, jk = r[order[2]], order[2]          # smallest selected
    rk1, jk1 = r[order[3]], order[3]        # largest unselected
    expect = torch.zeros(16, dtype=torch.float64)
    for c in range(8):
        theta, j = (rk1, jk1) if m[c] > 0 else (rk, jk)
        dens = math.exp(-abs(float(theta - u[c])) / 0.4) / 0.8
        sel = dens * (float(gw[c] * v[c]) - float(gw[j] * v[j]))
        expect[idx[c]] = float(gw[c] * m[c]) + math.copysign(1.0, float(v[c])) * sel
    assert torch.allclose(g[0], expect, atol=1e-10)


def test_value_path_alone_with_zero_scale_is_the_hard_mask_gradient():
    torch.manual_seed(1)
    gate = gate_of(policy_support_scale=0.0)
    z = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 16, dtype=torch.float64)
    y, col = forward(gate, z, seed=2)
    (g,) = torch.autograd.grad((y * w).sum(), [z])
    assert torch.equal(g, w * (y != 0).to(w.dtype))
    # the density record is kept for the trainer, with zero weight
    assert float(col.records[0].log_prob.abs().max()) == 0.0


def test_first_order_scope_linearizes_with_the_hard_path_gradient():
    """Two gates in a chain: the first-order scope must equal the full scope
    when the downstream gate has no selection term, and differ otherwise."""
    torch.manual_seed(7)
    W = torch.randn(16, 16, dtype=torch.float64) * 0.5
    w = torch.randn(16, dtype=torch.float64)
    z0 = torch.randn(6, 16, dtype=torch.float64)

    def grad(scope, gamma2):
        g1 = gate_of(policy_rb_scope=scope)
        g2 = gate_of(policy_rb_scope=scope, policy_support_scale=gamma2)
        z = z0.clone().requires_grad_(True)
        col = PolicyCollector()
        gen = make_generator(CPU, 11)
        with policy_forward([g1, g2], PolicyForwardSettings(sample=True, collector=col,
                                                             generator=gen)):
            h = g2(g1(z) @ W)
        loss = (h * w).sum()
        if scope == "first_order":
            first_order_backward(loss, z)
            return z.grad.clone()
        (g,) = torch.autograd.grad(loss, [z])
        return g

    assert torch.allclose(grad("full", 0.0), grad("first_order", 0.0), atol=1e-12)
    assert not torch.allclose(grad("full", 1.0), grad("first_order", 1.0), atol=1e-6)


def test_scale_free_width_gives_no_gradient_along_a_common_rescaling():
    """With policy_width_gradient=through the selection is exactly invariant to
    multiplying every score by a constant, so the selection gradient has no
    component along z itself (the frozen-scale rule does)."""
    torch.manual_seed(4)
    z0 = torch.randn(1, 16, dtype=torch.float64)
    w = torch.randn(16, dtype=torch.float64)

    def radial(wg):
        gate = gate_of(policy_temperature_mode="relative_span", policy_width_gradient=wg,
                       temperature=0.5)
        z = z0.repeat(20_000, 1).clone().requires_grad_(True)
        y, _ = forward(gate, z, seed=3)
        mask = (y != 0).to(z.dtype)
        (g,) = torch.autograd.grad((y * w).sum(), [z])
        g_sel = g - w * mask                  # remove the value path
        return float((g_sel * z.detach()).sum(-1).mean())

    assert abs(radial("through")) < 1e-9
    assert abs(radial("frozen")) > 1e-3


@pytest.mark.parametrize("bad", [
    dict(policy_estimator="nope"),
    dict(policy_rb_scope="first_order"),                       # needs rao_blackwell
    dict(policy_estimator="rao_blackwell", policy_rb_scope="nope"),
    dict(policy_width_gradient="through"),                     # absolute width
    dict(policy_width_gradient="nope", policy_temperature_mode="relative_span"),
])
def test_invalid_estimator_settings_are_rejected(bad):
    kw = dict(enabled=True, n_features=64, k=8, j=8, surrogate_mode="laplace_policy",
              temperature=0.5)
    kw.update(bad)
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**kw)


def test_estimator_fields_under_another_mode_are_rejected():
    with pytest.raises(ValueError, match="policy_"):
        ActivationBottleneckConfig(enabled=True, n_features=64, k=8, j=8,
                                   surrogate_mode="hard", policy_estimator="rao_blackwell")


def make_data(tmp_path, vocab=64):
    import json
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for name, n in (("train", 60_000), ("val", 8_000)):
        arr = np.memmap(d / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(n,))
        arr[:] = rng.integers(0, vocab, size=n).astype(np.uint16)
        arr.flush()
        del arr
    with open(d / "meta.json", "w") as f:
        json.dump({"vocab_size": vocab, "eos_id": 0, "tokenizer": "fake",
                   "train_tokens": 60_000}, f)
    return str(d)


@pytest.mark.parametrize("scope,wg", [("full", "frozen"), ("first_order", "through")])
def test_training_runs_with_the_rao_blackwell_estimator(tmp_path, scope, wg):
    cfg = Config()
    cfg.data.data_dir = make_data(tmp_path)
    cfg.data.seq_len = 32
    cfg.model = ModelConfig(vocab_size=64, max_seq_len=32, n_layers=2, d_model=32, n_heads=4)
    cfg.train.batch_size = 8
    cfg.train.micro_batch_size = 4
    cfg.train.max_steps = 6
    cfg.train.warmup_steps = 2
    cfg.train.log_every_steps = 2
    cfg.train.validate_every_steps = 3
    cfg.train.val_batches = 2
    cfg.train.checkpoint_every_steps = 6
    cfg.train.dtype = "float32"
    cfg.train.device = "cpu"
    cfg.train.tensorboard = False
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.run_name = f"rb_{scope}"
    cfg.activation_bottleneck = ActivationBottleneckConfig(
        enabled=True, n_features=128, k=16, j=48, layers="all", placement="residual_out",
        surrogate_mode="laplace_policy", temperature=0.5,
        policy_temperature_mode="relative_span", policy_estimator="rao_blackwell",
        policy_rb_scope=scope, policy_width_gradient=wg, code_residual=True)
    summary = train(cfg)
    assert math.isfinite(summary["val/ce"]) and math.isfinite(summary["train/ce"])
    assert summary["policy/support_term_diag"] == 0.0
    assert "bottleneck/policy_rb_score_grad_rms" in summary
    assert math.isfinite(summary["bottleneck/policy_rb_score_grad_rms"])
