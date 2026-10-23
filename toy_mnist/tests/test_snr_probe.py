import numpy as np
import torch

from toy_mnist.config import N_PIXELS, load_config
from toy_mnist.dust import make_generator, noise_scale
from toy_mnist.model import ToyMLP
from toy_mnist.snr_probe import fixed_batch, list_checkpoints, snr_at_state
from toy_mnist.tests.helpers import synthetic_data


def _brute_force(model, x, y, P, sigma, seed):
    """Every draw stored: the module docstring's formulas verbatim."""
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        u = model.gate_input(x)
        B, N = u.shape
        gen = make_generator(x.device, seed)
        a = torch.randn((P, B, N), generator=gen)
        losses = torch.stack([F.cross_entropy(model.forward_from_gate(u + sigma * a[i]), y,
                                              reduction="none") for i in range(P)])  # [P, B]
        r = losses.mean(0, keepdim=True) - losses                     # R~ - R
        z = -r.unsqueeze(-1) * a / sigma                               # [P, B, N]
        w = torch.exp(model.rescale.log_scale) * x
        z_u, z_g = z.mean(1), (z * w).mean(1)                          # [P, N]
        out = {}
        for name, zz in (("u", z_u), ("g", z_g)):
            mu = zz.mean(0)
            nu = zz.std(0, unbiased=True)
            out[name] = (mu, nu, P ** 0.5 * mu.abs() / nu)
        return out


def test_two_pass_probe_matches_the_stored_draws():
    torch.manual_seed(0)
    cfg = load_config(values={"rescale": {"enabled": True}, "model": {"hidden": 8},
                              "topk": {"enabled": True, "k": 6, "surrogate_mode": "dust",
                                       "dust_population": 16}})
    model = ToyMLP(cfg)
    with torch.no_grad():
        model.rescale.log_scale.normal_(std=0.3)
    x = torch.rand(4, N_PIXELS)
    y = torch.randint(0, 10, (4,))
    out = snr_at_state(model, x, y, population=16, sigma=0.2, sigma_mode="absolute",
                       noise_seed=3, chunk=5)  # uneven chunks: 5, 5, 5, 1
    ref = _brute_force(model, x, y, 16, 0.2, 3)
    for name in ("u", "g"):
        mu, nu, snr = ref[name]
        assert torch.allclose(out[f"mu_{name}"], mu, atol=1e-6, rtol=1e-4)
        assert torch.allclose(out[f"nu_{name}"], nu, atol=1e-6, rtol=1e-4)
        ok = torch.isfinite(out[f"snr_{name}"])
        assert torch.allclose(out[f"snr_{name}"][ok], snr[ok], atol=1e-4, rtol=1e-3)
    assert torch.isfinite(out["snr_u"]).all()   # every coordinate is perturbed
    assert out["ce_perturbed"] > 0 and out["ce_clean"] > 0
    # a pixel that is 0 in every sample: the g weighting has no signal and no noise
    x0 = x.clone()
    x0[:, 5] = 0.0
    out0 = snr_at_state(model, x0, y, population=16, sigma=0.2, sigma_mode="absolute",
                        noise_seed=3, chunk=16)
    assert torch.isnan(out0["snr_g"][5]) and torch.isfinite(out0["snr_u"][5])


def test_fixed_batch_and_checkpoint_listing(tmp_path):
    data = synthetic_data(n_train=100)
    idx = fixed_batch(data, 16, seed=0)
    assert idx.shape == (16,) and len(set(idx.tolist())) == 16
    assert torch.equal(idx, fixed_batch(data, 16, seed=0))
    for n in (0, 7, 500, 3):
        (tmp_path / f"ckpt_step{n}.pt").write_bytes(b"x")
    (tmp_path / "final.pt").write_bytes(b"x")
    assert [n for n, _ in list_checkpoints(str(tmp_path), 5000)] == [0, 3, 7, 500, 5000]
    (tmp_path / "ckpt_step5000.pt").write_bytes(b"x")
    assert [n for n, _ in list_checkpoints(str(tmp_path), 5000)] == [0, 3, 7, 500, 5000]
