"""Permutation ablation of the rblapsum surrogate signal.

rho fraction of each row's Top(K+J) pool has its dL/dp values shuffled before
the kernel weighting; positions keep their own kappa, the zero-sum correction
applies to the permuted signal, and the hard task path is never touched.
"""

import pytest
import torch

from wsparse.bottleneck import AdaptiveLapSumTopKGate
from wsparse.bottleneck.rblapsum import permute_fraction
from wsparse.config import ActivationBottleneckConfig


def make_gate(rho, mode="through_rank_kappa", supp=1.0, k=3, j=4, n=16):
    return AdaptiveLapSumTopKGate(
        n_features=n, k=k, j=j, n_eff=3.0, selection_mode="abs_topk",
        surrogate_mode="rblapsum", rblapsum_boundary_grad_mode=mode,
        rblapsum_boundary_floor=0.0, rblapsum_temperature=1.0,
        rblapsum_support_scale=supp,
        rblapsum_rho_random_perm_prob_grad=rho, log_diagnostics=True)


def grad_of(gate, a, up, train=True):
    a = a.clone().requires_grad_(True)
    gate.train(train)
    gate(a).backward(up)
    return a.grad.clone()


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #


def test_permute_fraction_preserves_multiset_and_moves_only_subset():
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float64)
    for rho, n_sel in ((0.5, 16), (1.0, 32)):
        y = permute_fraction(x, rho)
        # per-row multiset preserved exactly
        assert torch.equal(x.sort(-1).values, y.sort(-1).values)
        # at most n_sel positions changed per row
        changed = (x != y).sum(-1)
        assert int(changed.max()) <= n_sel
        assert int(changed.float().mean()) > 0  # it does something
    # rho=0 and tiny subsets are exact no-ops
    assert permute_fraction(x, 0.0) is x
    assert permute_fraction(x, 0.04) is x  # round(0.04*32)=1 < 2


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_rho_zero_bitwise_identical():
    torch.manual_seed(1)
    a = torch.randn(6, 16, dtype=torch.float64)
    up = torch.randn(6, 16, dtype=torch.float64)
    torch.manual_seed(7)
    g0 = grad_of(make_gate(0.0), a, up)
    torch.manual_seed(7)
    g_ref = grad_of(make_gate(0.0), a, up)
    assert torch.equal(g0, g_ref)


def test_perm_changes_score_path_not_task_path():
    torch.manual_seed(2)
    a = torch.randn(6, 16, dtype=torch.float64)
    up = torch.randn(6, 16, dtype=torch.float64)
    g0 = grad_of(make_gate(0.0), a, up)
    g1 = grad_of(make_gate(1.0), a, up)
    assert not torch.equal(g0, g1)
    # with the surrogate scaled to zero the two agree exactly: the hard task
    # path is untouched by the permutation
    h0 = grad_of(make_gate(0.0, supp=0.0), a, up)
    h1 = grad_of(make_gate(1.0, supp=0.0), a, up)
    assert torch.equal(h0, h1)


def test_zero_sum_survives_full_permutation():
    torch.manual_seed(3)
    a = torch.randn(8, 16, dtype=torch.float64)
    up = torch.randn(8, 16, dtype=torch.float64)
    g = make_gate(1.0)
    dz = grad_of(g, a, up)
    scores = a.abs()
    cs, ci = torch.topk(scores, g.m, dim=-1, largest=True, sorted=True)
    active = torch.zeros_like(cs)
    active[..., :g.k] = 1.0
    z_c = torch.gather(a, -1, ci)
    u_c = torch.gather(up, -1, ci)
    g_s = z_c.sign() * (torch.gather(dz, -1, ci) - u_c * active)
    assert g_s.sum(-1).abs().max() < 1e-10


def test_perm_is_off_in_eval():
    torch.manual_seed(4)
    a = torch.randn(6, 16)
    g1, g0 = make_gate(1.0), make_gate(0.0)
    g1.eval(), g0.eval()
    with torch.no_grad():
        assert torch.equal(g1(a), g0(a))


def test_config_validation():
    ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum",
                               selection_mode="abs_topk",
                               rblapsum_rho_random_perm_prob_grad=0.5)
    with pytest.raises(ValueError, match="must be in"):
        ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum",
                                   selection_mode="abs_topk",
                                   rblapsum_rho_random_perm_prob_grad=1.5)
    with pytest.raises(ValueError, match="rblapsum"):
        ActivationBottleneckConfig(enabled=True, surrogate_mode="rblapsum_sf",
                                   selection_mode="abs_topk",
                                   rblapsum_rho_random_perm_prob_grad=0.5)
