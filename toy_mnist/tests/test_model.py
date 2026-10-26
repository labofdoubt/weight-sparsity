import torch

from toy_mnist.config import N_PIXELS, load_config
from toy_mnist.model import AbsTopKGate, DiagonalRescale, ToyMLP


def test_baseline_shapes_and_loss():
    cfg = load_config(values={"model": {"hidden": 8}})
    model = ToyMLP(cfg)
    assert model.rescale is None and model.gate is None
    x = torch.rand(4, N_PIXELS)
    y = torch.randint(0, 10, (4,))
    logits, ce = model(x, y)
    assert logits.shape == (4, 10) and ce.ndim == 0
    # z = W2 relu(W1 x), nothing else
    ref = model.fc2(torch.relu(model.fc1(x)))
    assert torch.equal(logits, ref)


def test_rescale_is_the_identity_at_init_and_scales_by_exp_g():
    r = DiagonalRescale(5)
    x = torch.randn(3, 5)
    assert torch.equal(r(x), x)
    with torch.no_grad():
        r.log_scale.copy_(torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5]))
    assert torch.allclose(r(x), x * torch.exp(r.log_scale))


def test_gate_keeps_the_k_largest_magnitudes_and_masks_the_backward():
    torch.manual_seed(0)
    gate = AbsTopKGate(10, k=3, j=2)
    assert gate.m == 5
    u = torch.randn(4, 10, requires_grad=True)
    out = gate(u)
    top = torch.topk(u.detach().abs(), 3, dim=-1).indices
    mask = torch.zeros(4, 10).scatter(-1, top, 1.0)
    assert torch.equal(out, u.detach() * mask)
    assert int((out != 0).sum(-1).max()) == 3
    out.sum().backward()
    assert torch.equal(u.grad, mask)  # d(sum out)/du = mask: no gradient off the support


def test_gate_gradient_reaches_g_through_active_entries_only():
    torch.manual_seed(1)
    cfg = load_config(values={"rescale": {"enabled": True},
                              "topk": {"enabled": True, "k": 5, "j": 3}, "model": {"hidden": 8}})
    model = ToyMLP(cfg)
    x = torch.rand(6, N_PIXELS)
    y = torch.randint(0, 10, (6,))
    _, ce = model(x, y)
    ce.backward()
    g_grad = model.rescale.log_scale.grad
    active = model.gate.seen  # features active in at least one sample
    assert int(active.sum()) <= 6 * 5
    assert torch.equal(g_grad[~active], torch.zeros_like(g_grad[~active]))
    assert g_grad[active].abs().sum() > 0


def test_trivial_gate_and_usage_statistics():
    gate = AbsTopKGate(6, k=6)
    assert gate.trivial
    u = torch.randn(2, 6)
    assert torch.equal(gate(u), u)
    gate = AbsTopKGate(6, k=2)
    gate.train()
    gate(torch.randn(8, 6))
    assert int(gate.usage_n) == 1 and abs(float(gate.usage.sum()) - 2.0) < 1e-6
    gate.eval()
    gate(torch.randn(8, 6))
    assert int(gate.usage_n) == 1  # eval forwards do not count


def test_capture_records_the_gate_input_and_forward_from_gate_matches():
    cfg = load_config(values={"rescale": {"enabled": True},
                              "topk": {"enabled": True, "k": 4}, "model": {"hidden": 8}})
    model = ToyMLP(cfg)
    with torch.no_grad():
        model.rescale.log_scale.normal_()
    x = torch.rand(3, N_PIXELS)
    records = []
    with model.gate.capture(records):
        logits, _ = model(x)
    assert len(records) == 1
    assert torch.equal(records[0], model.gate_input(x))
    assert torch.equal(model.forward_from_gate(records[0]), logits)
    assert model.gate._dust_capture is None


def test_pre_linear_layer_and_the_two_gate_sites():
    torch.manual_seed(3)
    x = torch.rand(5, N_PIXELS)
    y = torch.randint(0, 10, (5,))
    # site input: x' = W0 AbsTopK(x)
    cfg = load_config(values={"model": {"hidden": 8, "pre_width": 12},
                              "topk": {"enabled": True, "k": 3, "site": "input"}})
    m = ToyMLP(cfg)
    assert m.pre.weight.shape == (12, N_PIXELS) and m.pre.bias is None
    assert m.gate.n_features == N_PIXELS and m.fc1.in_features == 12
    logits, _ = m(x)
    ref = m.fc2(torch.relu(m.fc1(m.pre(m.gate(x)))))
    assert torch.equal(logits, ref)
    assert torch.equal(m.gate_input(x), x)
    # site pre_linear: x' = AbsTopK(W0 x); the gate has pre_width units
    cfg = load_config(values={"model": {"hidden": 8, "pre_width": 12},
                              "topk": {"enabled": True, "k": 3, "site": "pre_linear"}})
    m = ToyMLP(cfg)
    assert m.gate.n_features == 12 and m.fc1.in_features == 12
    logits, ce = m(x, y)
    u = m.pre(x)
    assert torch.equal(m.gate_input(x), u)
    assert torch.equal(logits, m.fc2(torch.relu(m.fc1(m.gate(u)))))
    assert int((m.gate(u) != 0).sum(-1).max()) == 3
    # the hard backward reaches W0 through the active units only: a row of W0
    # whose unit is never selected gets no gradient
    ce.backward()
    active = m.gate.seen
    rows = m.pre.weight.grad.abs().sum(-1)
    assert torch.equal(rows[~active], torch.zeros_like(rows[~active]))
    assert rows[active].sum() > 0
    # site pre_linear needs W0; k is bounded by the gated width
    import pytest
    with pytest.raises(ValueError, match="pre_width"):
        load_config(values={"topk": {"enabled": True, "k": 3, "site": "pre_linear"}})
    with pytest.raises(ValueError, match="topk.k"):
        load_config(values={"model": {"pre_width": 12},
                            "topk": {"enabled": True, "k": 13, "site": "pre_linear"}})
    # dust at site pre_linear needs no rescaling (W0 is the encoder); at site
    # input it still does
    cfg = load_config(values={"model": {"pre_width": 12},
                              "topk": {"enabled": True, "k": 3, "site": "pre_linear",
                                       "surrogate_mode": "dust"}})
    assert cfg.topk.resolved_j(12) == 9
    with pytest.raises(ValueError, match="rescale.enabled"):
        load_config(values={"model": {"pre_width": 12},
                            "topk": {"enabled": True, "k": 3, "site": "input",
                                     "surrogate_mode": "dust"}})
