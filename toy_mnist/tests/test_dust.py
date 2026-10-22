import math

import pytest
import torch

from toy_mnist.config import N_PIXELS, load_config
from toy_mnist.dust import (DustRunningStats, DustTrainingState, dust_estimate, dust_micro_step,
                            dust_step_kind, make_generator, noise_scale, pool_indices)
from toy_mnist.model import ToyMLP


def test_step_schedule_starts_each_period_with_the_hard_steps():
    assert [dust_step_kind(s, [1, 1]) for s in range(4)] == ["hard", "dust", "hard", "dust"]
    assert [dust_step_kind(s, [1, 3]) for s in range(5)] == ["hard", "dust", "dust", "dust", "hard"]
    assert [dust_step_kind(s, [0, 1]) for s in range(3)] == ["dust"] * 3
    assert [dust_step_kind(s, [2, 1]) for s in range(4)] == ["hard", "hard", "dust", "hard"]
    with pytest.raises(ValueError):
        dust_step_kind(0, [1, 0])


def test_linear_loss_estimate_is_the_gradient_on_the_pool_only():
    torch.manual_seed(2)
    B, N, M = 3, 12, 6
    u = torch.randn(B, N)
    w = torch.randn(N)
    cand = pool_indices(u, M)
    assert torch.equal(cand, torch.topk(u.abs(), M, dim=-1).indices)
    gen = torch.Generator().manual_seed(0)
    loss_fn = lambda up: (up * w).sum(-1)                                    # noqa: E731
    for mode in ("absolute", "relative_rms"):
        sigma_t = noise_scale(mode, 0.3, u, cand)
        assert sigma_t.shape == (B, 1)
        ghat, st = dust_estimate(u, cand, sigma_t, draws=8000, loss_fn=loss_fn,
                                 generator=gen, chunk=200)
        expected = w[cand] * (1.0 - 1.0 / 8000)
        assert torch.allclose(ghat, expected, atol=0.15), (ghat - expected).abs().max()
        assert st["draws"].item() == 8000
    # the draw-mean baseline: E g^ = (1 - 1/K) grad, here K = 2
    acc = torch.zeros(B, M)
    reps = 3000
    sigma_t = noise_scale("absolute", 0.3, u, cand)
    for _ in range(reps):
        g, _ = dust_estimate(u, cand, sigma_t, draws=2, loss_fn=loss_fn, generator=gen, chunk=2)
        acc += g
    assert torch.allclose(acc / reps, 0.5 * w[cand], atol=0.15)


def test_estimate_matches_the_main_repository_with_one_token():
    """The same formulas as wsparse.bottleneck.dust on [B, T=1, N] inputs and
    the same noise stream (skipped where the main package is not importable)."""
    main = pytest.importorskip("wsparse.bottleneck.dust")
    torch.manual_seed(5)
    B, N, M, K = 4, 20, 7, 16
    u = torch.randn(B, N)
    w = torch.randn(N)
    cand = pool_indices(u, M)
    cand_main = main.pool_indices(u.unsqueeze(1), "abs_topk", M)
    assert torch.equal(cand_main.squeeze(1), cand)
    for mode in ("absolute", "relative_rms"):
        sigma = noise_scale(mode, 0.2, u, cand)
        sigma_main = main.noise_scale(mode, 0.2, u.unsqueeze(1), cand_main)
        assert torch.allclose(sigma, sigma_main.squeeze(1))
        loss = lambda up: torch.tanh((up * w).sum(-1))                        # noqa: E731
        loss_main = lambda zp: torch.tanh((zp * w).sum(-1))                   # noqa: E731
        g1, s1 = dust_estimate(u, cand, sigma, K, loss, torch.Generator().manual_seed(7), chunk=4)
        g2, s2 = main.dust_estimate(u.unsqueeze(1), cand_main, sigma_main, K, 0.0,
                                    loss_main, torch.Generator().manual_seed(7), chunk=4)
        assert torch.allclose(g1, g2.squeeze(1), atol=1e-5)
        assert torch.allclose(s1["ell_sum"], s2["ell_sum"])
        assert torch.allclose(s1["reward_sq_sum"], s2["reward_sq_sum"], atol=1e-4)


def _model(k, j=0, hidden=16, seed=0, n=N_PIXELS):
    torch.manual_seed(seed)
    cfg = load_config(values={"rescale": {"enabled": True},
                              "topk": {"enabled": True, "k": k, "j": j},
                              "model": {"hidden": hidden}})
    model = ToyMLP(cfg)
    with torch.no_grad():
        model.rescale.log_scale.normal_(std=0.3)
    return cfg, model


def _state(cfg, **over):
    for k, v in over.items():
        setattr(cfg.topk, k, v)
    cfg.topk.surrogate_mode = "dust"
    return DustTrainingState(cfg.topk, torch.device("cpu"), seed=11)


def test_micro_step_encoder_and_backprop_targets_give_the_same_gradient_on_g_only():
    cfg, model = _model(k=16, j=16)
    x = torch.rand(8, N_PIXELS)
    y = torch.randint(0, 10, (8,))
    grads = {}
    for target in ("encoder", "backprop"):
        st = _state(cfg, dust_population=6, dust_sigma=0.1, dust_target=target, dust_chunk_draws=3)
        model.zero_grad(set_to_none=True)
        out = dust_micro_step(model, x, y, st, want_cosine=True)
        assert model.fc1.weight.grad is None and model.fc2.weight.grad is None
        g = model.rescale.log_scale.grad
        assert g is not None and torch.isfinite(g).all()
        grads[target] = g.clone()
        assert math.isfinite(out["ce_clean"]) and 0.0 <= out["acc_clean"] <= 1.0
        assert out["ce_perturbed"] == pytest.approx(out["ce_clean"] + out["ce_gap"], abs=1e-5)
        assert out["reward_rms"] >= 0 and out["ghat_rms"] > 0
        assert out["cosine"] is not None and -1.0 <= out["cosine"] <= 1.0
    assert torch.allclose(grads["encoder"], grads["backprop"], atol=1e-6, rtol=1e-4)
    # nonzero only on pool entries of some sample: the pool of the batch
    u = model.gate_input(x)
    pool = torch.zeros(N_PIXELS, dtype=torch.bool)
    pool[pool_indices(u, model.gate.m).reshape(-1)] = True
    assert torch.equal(grads["encoder"][~pool], torch.zeros_like(grads["encoder"][~pool]))


def test_trivial_gate_estimate_aligns_with_the_smooth_gradient():
    """k = N: the gate is the identity, the loss is smooth in u, and with a
    small sigma and many draws the estimate follows the true gradient of g."""
    cfg, model = _model(k=N_PIXELS, j=0, hidden=16)
    x = torch.rand(16, N_PIXELS)
    y = torch.randint(0, 10, (16,))
    _, ce = model(x, y)
    (true,) = torch.autograd.grad(ce, [model.rescale.log_scale])
    st = _state(cfg, dust_population=20000, dust_sigma=0.01, dust_chunk_draws=500)
    model.zero_grad(set_to_none=True)
    out = dust_micro_step(model, x, y, st, want_cosine=True)
    est = model.rescale.log_scale.grad
    cos = float((est * true).sum() / (est.norm() * true.norm()))
    assert cos > 0.9, cos
    assert out["cosine"] == pytest.approx(cos, abs=1e-5)


def test_grad_scale_and_generator_reproducibility():
    cfg, model = _model(k=8, j=8)
    x = torch.rand(5, N_PIXELS)
    y = torch.randint(0, 10, (5,))
    outs = []
    for scale in (1.0, 3.0):
        st = _state(cfg, dust_population=4, dust_grad_scale=scale, dust_sigma=0.2)
        model.zero_grad(set_to_none=True)
        o = dust_micro_step(model, x, y, st)
        outs.append((model.rescale.log_scale.grad.clone(), o["ghat_rms"]))
    assert torch.allclose(outs[1][0], 3.0 * outs[0][0], atol=1e-6, rtol=1e-5)
    assert outs[0][1] == pytest.approx(outs[1][1])  # ghat_rms is the unscaled estimate
    st = _state(cfg, dust_population=4, dust_sigma=0.2, dust_grad_scale=1.0)
    model.zero_grad(set_to_none=True)
    dust_micro_step(model, x, y, st)  # a fresh generator with the same seed: the same noise
    assert torch.equal(model.rescale.log_scale.grad, outs[0][0])


def test_state_round_trip_continues_the_noise_stream():
    cfg, model = _model(k=8, j=8)
    st = _state(cfg, dust_population=4)
    st.dust_steps = 7
    torch.randn(10, generator=st.generator)  # advance the stream
    payload = st.state_dict()
    st2 = _state(cfg, dust_population=4)
    st2.load_state_dict(payload)
    assert st2.dust_steps == 7
    assert torch.equal(torch.randn(10, generator=st2.generator),
                       torch.randn(10, generator=st.generator))
    with pytest.raises(ValueError):
        st2.load_state_dict({"version": 0, "dust_steps": 0,
                             "generator_state": payload["generator_state"]})


def test_running_stats_average_per_step_and_report_cosine_when_present():
    s = DustRunningStats()
    assert s.metrics() == {"dust/steps": 0.0}
    base = {"ce_clean": 1.0, "ce_perturbed": 1.5, "ce_gap": 0.5, "reward_rms": 0.1,
            "ghat_rms": 2.0, "cosine": None}
    s.add(base)
    s.add({**base, "ce_perturbed": 2.5, "ce_gap": 1.5, "cosine": 0.4})
    s.add_step(3.0)
    m = s.metrics()
    assert m["dust/steps"] == 1.0 and m["dust/ce_gap"] == pytest.approx(1.0)
    assert m["dust/cosine"] == pytest.approx(0.4) and m["dust/grad_norm"] == 3.0
