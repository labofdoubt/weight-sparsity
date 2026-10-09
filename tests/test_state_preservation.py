"""analysis/state_preservation.py: the ridge map and the transport."""

import importlib.util
import os
import types

import torch

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.config import ActivationBottleneckConfig, ModelConfig
from wsparse.model import build_model

_spec = importlib.util.spec_from_file_location(
    "state_preservation",
    os.path.join(os.path.dirname(__file__), "..", "analysis", "state_preservation.py"))
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def test_fit_ridge_recovers_a_linear_map_and_scores_the_constant_at_one():
    torch.manual_seed(0)
    n, N = 2000, 24
    y = torch.randn(n, N) * torch.linspace(0.5, 2.0, N)
    A = torch.randn(N, N) / N ** 0.5
    x = y @ A + 0.3 + 0.01 * torch.randn(n, N)
    tr, va = slice(0, 1500), slice(1500, 2000)
    W, mx, my, lam, table = sp.fit_ridge(x[tr], y[tr], x[va], y[va], [1e-6, 1e-4, 1e-2, 1.0])
    val_errs = [t[2] for t in table]
    assert min(val_errs) == dict((t[0], t[2]) for t in table)[lam]  # the chosen lambda is the best on validation
    pred = (x[va].double() - mx) @ W + my
    err = float(((pred - y[va]) ** 2).sum()) / float(((y[va] - my) ** 2).sum())
    assert err < 0.01                      # the map is recovered up to the noise
    const = float(((y[va] - my) ** 2).sum(1).mean()) / float(((y[va] - my) ** 2).sum(1).mean())
    assert const == 1.0


def test_transport_matches_the_model_with_block_updates_off():
    """With attention and MLP returning zero, the model's own forward from
    bottleneck s to e is exactly the transport."""
    torch.manual_seed(0)
    model = build_model(ModelConfig(vocab_size=97, max_seq_len=16, n_layers=4,
                                    d_model=32, n_heads=4))
    cfg = ActivationBottleneckConfig(enabled=True, n_features=64, k=6, j=10, layers="all",
                                     surrogate_mode="rblapsum", selection_mode="abs_topk",
                                     placement="residual_out", bias=False, post_norm=True)
    apply_activation_bottleneck(model, cfg, max_steps=10)
    model.eval()
    mods = sp.find_bottlenecks(model, types.SimpleNamespace(activation_bottleneck=cfg))
    idx = torch.randint(0, 97, (3, 16))
    s, e = 1, 3
    # the start code with the full model
    grabbed = {}
    h = mods[s].gate.register_forward_hook(lambda m, i, o: grabbed.__setitem__("c", o.detach()))
    with torch.no_grad():
        model(idx)
    h.remove()
    c_s = grabbed["c"]
    ct = sp.transport(mods, c_s, s, e)
    # the reference: run blocks s+1..e of the model with Delta = 0
    for blk in model.blocks[s + 1:e + 1]:
        blk.attn.forward = lambda x, *a, **k: torch.zeros_like(x)
        blk.mlp.forward = lambda x, *a, **k: torch.zeros_like(x)
    got = {}
    h = mods[e].gate.register_forward_hook(lambda m, i, o: got.__setitem__("c", o.detach()))
    with torch.no_grad():
        x = mods[s].post_norm(mods[s].decode(c_s))
        for blk in model.blocks[s + 1:e + 1]:
            x = blk(x)
    h.remove()
    assert torch.allclose(ct, got["c"], atol=1e-6)
    assert (ct != 0).sum(-1).eq(cfg.k).all()
