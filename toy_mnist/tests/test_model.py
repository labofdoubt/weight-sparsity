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
