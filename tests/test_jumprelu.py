"""jumprelu: hard margin-pool forward, rectangular-kernel gradients to theta only."""

import math

import pytest
import torch

from wsparse.bottleneck import (
    AdaptiveLapSumTopKGate,
    apply_activation_bottleneck,
    default_log_theta,
    jumprelu_count,
    jumprelu_forward,
    rect_kernel,
)
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model


def make_gate(**kw):
    cfg = dict(n_features=16, k=3, j=4, n_eff=3.0, selection_mode="abs_topk",
               surrogate_mode="jumprelu", jumprelu_kernel_width=0.5,
               jumprelu_count_coef=1.0, log_diagnostics=True)
    cfg.update(kw)
    return AdaptiveLapSumTopKGate(**cfg)


def bn_cfg(**kw):
    cfg = dict(enabled=True, n_features=64, k=8, j=24, n_eff=6.0, layers="all",
               surrogate_mode="jumprelu", selection_mode="abs_topk",
               placement="residual_out", bias=False)
    cfg.update(kw)
    return ActivationBottleneckConfig(**cfg)


def set_theta(gate, values):
    with torch.no_grad():
        gate.log_theta.copy_(torch.tensor(values, dtype=torch.float32).log())
        gate.theta_calibrated.fill_(1)   # explicit thetas: skip first-forward calibration


# --------------------------------------------------------------------------- #
# selection and forward
# --------------------------------------------------------------------------- #


def test_candidates_ranked_by_margin_not_score():
    gate = make_gate(n_features=6, k=1, j=1)
    gate.train()
    # feature 0 has the LARGEST score and would be active if selected
    # (|a|=5 > theta=4.9), but its margin 0.1 loses to features 1 and 2
    set_theta(gate, [4.9, 0.1, 0.1, 10.0, 10.0, 10.0])
    a = torch.tensor([[5.0, 4.0, 3.0, 0.2, 0.1, 0.05]], requires_grad=True)
    y = gate(a)
    assert y[0, 0].item() == 0.0                    # score-ranking would keep it
    assert y[0, 1].item() == 4.0 and y[0, 2].item() == 3.0
    # ... and the theta gradient reaches only the margin-selected pool
    y.sum().backward()
    assert gate.log_theta.grad[3:].abs().sum().item() == 0.0
    assert gate.log_theta.grad[0].abs().item() == 0.0   # not in the pool of 2


def test_at_most_k_plus_j_participate():
    torch.manual_seed(0)
    gate = make_gate(n_features=32, k=3, j=4, jumprelu_theta_init=0.5,
                     jumprelu_kernel_width=5.0)   # huge window: every candidate in it
    gate.train()
    a = torch.randn(4, 32, requires_grad=True)
    y = gate(a)
    (y * torch.randn_like(y)).sum().backward()
    assert int((y != 0).sum(-1).max()) <= gate.m
    # theta gradient support: at most m features per row can contribute; with
    # 4 rows the union is bounded by 4*m
    assert int((gate.log_theta.grad != 0).sum()) <= 4 * gate.m


def test_forward_is_exactly_z_times_heaviside_on_the_pool():
    torch.manual_seed(1)
    gate = make_gate(n_features=24, k=4, j=6)
    gate.train()
    theta = torch.rand(24).double() + 0.3
    set_theta(gate, theta.tolist())
    a = (torch.randn(5, 24)).requires_grad_(True)
    y = gate(a)
    margin = a.detach().abs() - theta.float()
    pool = torch.zeros_like(margin, dtype=torch.bool)
    pool.scatter_(-1, margin.topk(gate.m, dim=-1).indices, True)
    expected = a.detach() * ((margin > 0) & pool).float()
    assert torch.equal(y.detach(), expected)


def test_rectangular_kernel_values():
    T = 0.25
    m = torch.tensor([0.0, 0.24999, -0.24999, 0.25, -0.25, 1.0], dtype=torch.float64)
    k = rect_kernel(m, T)
    assert k[0] == k[1] == k[2] == pytest.approx(1.0 / (2 * T))
    assert k[3] == k[4] == k[5] == 0.0


def test_train_and_eval_forwards_are_identical():
    torch.manual_seed(2)
    gate = make_gate(n_features=32, k=4, j=4)
    a = torch.randn(3, 32)
    gate.train()
    y_train = gate(a.clone())
    gate.eval()
    y_eval = gate(a.clone())
    assert torch.equal(y_train, y_eval)


# --------------------------------------------------------------------------- #
# backward rules
# --------------------------------------------------------------------------- #


def test_boundary_gradient_reaches_theta_but_not_z():
    gate = make_gate(n_features=6, k=1, j=2, jumprelu_kernel_width=0.5)
    gate.train()
    # feature 1: inactive candidate NEAR its boundary (margin -0.2, in window)
    set_theta(gate, [0.5, 3.2, 10.0, 10.0, 10.0, 10.0])
    a = torch.tensor([[5.0, 3.0, 0.1, 0.05, 0.02, 0.01]], requires_grad=True)
    y = gate(a)
    assert y[0, 1].item() == 0.0                       # inactive: below threshold
    g = torch.zeros_like(y); g[0, 1] = 1.0             # upstream only on feature 1
    y.backward(g)
    assert gate.log_theta.grad[1].abs().item() > 0.0   # theta moves
    assert a.grad[0, 1].item() == 0.0                  # z receives nothing


def test_active_candidate_gets_plain_task_gradient_through_z():
    gate = make_gate(n_features=6, k=2, j=2)
    gate.train()
    set_theta(gate, [1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    a = torch.tensor([[4.0, -3.0, 0.2, 0.1, 0.05, 0.02]], requires_grad=True)
    y = gate(a)
    assert y[0, 0].item() == 4.0 and y[0, 1].item() == -3.0
    g = torch.randn_like(y)
    y.backward(g)
    # dy/dz = H = 1 exactly: the upstream gradient passes through unchanged,
    # signs included (the |a| score is detached and contributes nothing)
    assert a.grad[0, 0].item() == pytest.approx(g[0, 0].item())
    assert a.grad[0, 1].item() == pytest.approx(g[0, 1].item())


def test_outside_pool_gets_no_gradient_at_all():
    torch.manual_seed(3)
    gate = make_gate(n_features=32, k=3, j=3, jumprelu_theta_init=1.0,
                     jumprelu_kernel_width=50.0)   # window covers everything
    gate.train()
    a = torch.randn(2, 32, requires_grad=True)
    y = gate(a)
    margin = a.detach().abs() - gate.log_theta.detach().exp()
    pool_idx = margin.topk(gate.m, dim=-1).indices
    outside = torch.ones(32, dtype=torch.bool)
    outside[pool_idx.reshape(-1).unique()] = False
    (y * torch.randn_like(y)).sum().backward()
    assert gate.log_theta.grad[outside].abs().sum().item() == 0.0
    assert a.grad[:, outside].abs().sum().item() == 0.0


# --------------------------------------------------------------------------- #
# count loss
# --------------------------------------------------------------------------- #


def test_count_loss_updates_theta_only():
    gate = make_gate(n_features=8, k=2, j=3, jumprelu_count_coef=1.0,
                     jumprelu_theta_init=1.0, jumprelu_kernel_width=2.0)
    gate.train()
    a = torch.tensor([[3.0, 2.5, 2.0, 1.5, 0.2, 0.1, 0.05, 0.02]],
                     requires_grad=True)
    gate(a)
    term = gate.take_count_loss()
    assert term is not None
    term.backward()
    assert gate.log_theta.grad is not None
    assert gate.log_theta.grad.abs().sum().item() > 0.0
    assert a.grad is None                     # z is not even in the count graph


def test_count_loss_direction():
    T = 1.0
    theta = torch.full((5,), 1.0, requires_grad=True, dtype=torch.float64)
    # all five candidates active and inside the window: L0 = 5 > k = 2
    score = torch.tensor([[1.5, 1.4, 1.3, 1.2, 1.1]], dtype=torch.float64)
    l0 = jumprelu_count(theta, score, T)
    assert float(l0) == 5.0
    ((2.0 - l0) ** 2).mean().backward()
    # descent step -lr*grad must INCREASE theta => gradient strictly negative
    assert (theta.grad < 0).all()

    theta2 = torch.full((5,), 1.0, requires_grad=True, dtype=torch.float64)
    score2 = torch.tensor([[0.9, 0.8, 0.7, 0.2, 0.1]], dtype=torch.float64)
    l02 = jumprelu_count(theta2, score2, T)
    assert float(l02) == 0.0                   # L0 < k = 2
    ((2.0 - l02) ** 2).mean().backward()
    assert (theta2.grad > 0).all()             # descent decreases theta


def test_count_loss_ignores_features_outside_the_pool():
    gate = make_gate(n_features=16, k=2, j=2, jumprelu_count_coef=1.0,
                     jumprelu_theta_init=1.0, jumprelu_kernel_width=50.0)
    gate.train()
    a = torch.zeros(1, 16); a[0, :6] = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.2, 1.1])
    gate(a.requires_grad_(True))
    gate.take_count_loss().backward()
    grad = gate.log_theta.grad
    assert grad[: gate.m].abs().sum().item() > 0.0
    assert grad[gate.m + 2:].abs().sum().item() == 0.0   # far outside the pool


def test_count_loss_steers_l0_toward_k():
    torch.manual_seed(4)
    gate = make_gate(n_features=64, k=8, j=24, jumprelu_count_coef=1.0,
                     jumprelu_theta_init=0.05, jumprelu_kernel_width=1.0)
    gate.train()
    a = torch.randn(16, 64)
    opt = torch.optim.Adam([gate.log_theta], lr=3e-2)
    def l0_now():
        with torch.no_grad():
            gate(a)
            return float(gate.diagnostics["active_count"])
    start = l0_now()
    for _ in range(300):
        opt.zero_grad()
        gate(a)
        gate.take_count_loss().backward()
        opt.step()
    end = l0_now()
    assert abs(end - gate.k) < abs(start - gate.k)
    assert abs(end - gate.k) < 2.0


# --------------------------------------------------------------------------- #
# logging, init, config, and non-interference
# --------------------------------------------------------------------------- #


def test_active_count_diagnostic_matches_hard_mask():
    torch.manual_seed(5)
    gate = make_gate(n_features=48, k=6, j=10)
    gate.train()
    a = torch.randn(7, 48)
    y = gate(a)
    manual = float((y != 0).sum(-1).float().mean())
    assert float(gate.diagnostics["active_count"]) == pytest.approx(manual)


def test_controller_exports_active_count():
    torch.manual_seed(6)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=32, n_layers=2,
                                    d_model=32, n_heads=4))
    cfg = bn_cfg(jumprelu_count_coef=0.5)
    ctl = apply_activation_bottleneck(model, cfg, max_steps=10)
    model.train()
    x = torch.randint(0, 97, (2, 16))
    _, loss = model(x, x)
    stats = ctl.stats(per_layer=True)
    assert "bottleneck/active_count" in stats
    assert any(k.startswith("bottleneck_active_count/") for k in stats)
    total, logs = ctl.count_loss()
    assert total is not None and "bottleneck/count_loss" in logs
    (loss + 0.5 * total).backward()   # the train-loop composition, in miniature
    g = model.blocks[0].residual_out_bottleneck.gate.log_theta.grad
    assert g is not None


def test_theta_init_placeholder_then_first_forward_calibration():
    gate = make_gate(n_features=1536, k=32, j=64)
    theta0 = float(gate.log_theta.exp()[0])
    p = 32 / 1536
    t = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(1.0 - p, dtype=torch.float64)))
    assert theta0 == pytest.approx(t, rel=1e-5)      # the pre-data placeholder
    assert theta0 == pytest.approx(math.exp(default_log_theta(32, 1536)), rel=1e-6)
    assert int(gate.theta_calibrated) == 0
    # first TRAINING forward re-centres theta at the batch's median k-th score,
    # whatever the score scale -- here deliberately not unit
    torch.manual_seed(7)
    gate.train()
    a = 0.05 * torch.randn(8, 1536)
    gate(a)
    kth = a.abs().topk(32, dim=-1).values[..., -1].reshape(-1).median()
    assert float(gate.log_theta.exp()[0]) == pytest.approx(float(kth), rel=1e-4)
    assert int(gate.theta_calibrated) == 1
    # so training does not begin gradient-dead even at this scale
    assert float(gate.diagnostics["in_window_frac"]) > 0.05
    # a second forward must NOT re-calibrate
    before = float(gate.log_theta.exp()[0])
    gate(2.0 * torch.randn(8, 1536))
    assert float(gate.log_theta.exp()[0]) == before
    # an eval-only gate never calibrates
    g2 = make_gate(n_features=64, k=4, j=4)
    g2.eval()
    g2(torch.randn(2, 64))
    assert int(g2.theta_calibrated) == 0
    # a numeric init pins theta and skips calibration entirely
    g3 = make_gate(n_features=64, k=4, j=4, jumprelu_theta_init=0.7)
    assert int(g3.theta_calibrated) == 1
    g3.train(); g3(torch.randn(2, 64))
    assert float(g3.log_theta.exp()[0]) == pytest.approx(0.7, rel=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        bn_cfg(selection_mode="topk")                      # needs abs_topk
    with pytest.raises(ValueError):
        bn_cfg(jumprelu_kernel_width=0.0)
    with pytest.raises(ValueError):
        bn_cfg(jumprelu_theta_init=-1.0)
    with pytest.raises(ValueError):
        bn_cfg(project_scale_gradient=True)                # LapSum-only knob
    with pytest.raises(ValueError):
        bn_cfg(inactive_grad_scale=0.5)                    # LapSum-only knob
    cfg = bn_cfg(jumprelu_count_coef=0.0)                  # loss is optional
    assert cfg.surrogate_mode == "jumprelu"


def test_lapsum_gates_gain_no_parameters_or_state():
    lap = AdaptiveLapSumTopKGate(n_features=16, k=3, j=4, n_eff=3.0,
                                 surrogate_mode="lapsum_scheduled")
    names = [n for n, _ in lap.named_parameters()]
    assert names == []                                     # gate had none before
    assert "log_theta" not in lap.state_dict()
    assert lap.take_count_loss() is None
    jr = make_gate()
    assert "log_theta" in jr.state_dict()                  # and jumprelu persists it
    assert "theta_calibrated" in jr.state_dict()           # calibration survives resume
    sd = jr.state_dict()
    jr2 = make_gate()
    jr2.load_state_dict(sd)
    assert torch.equal(jr2.log_theta, jr.log_theta)
