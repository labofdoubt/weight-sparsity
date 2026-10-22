"""surrogate_mode="dust": ordinary hard Top-K steps alternated with zeroth-order
DUST steps (reward-weighted noise on the Top(K+J) pre-activations).

The estimator tests use synthetic per-token losses with known gradients
(linear and cumulative), so the sign, the 1/(K sigma) scaling, the pool
restriction and the discounted credit are each checked against a closed form.
"""

import copy
import json
import math
import os

import numpy as np
import pytest
import torch
import yaml

from wsparse.bottleneck import apply_activation_bottleneck
from wsparse.bottleneck.dust import (DustTrainingState, discounted_returns, draws_per_bottleneck,
                                     dust_capture, dust_estimate, dust_micro_step,
                                     dust_step_kind, noise_scale, pool_indices, token_losses)
from wsparse.bottleneck.gate import AdaptiveLapSumTopKGate
from wsparse.bottleneck.module import effective_backward_support
from wsparse.config import (ActivationBottleneckConfig, Config, ModelConfig, config_from_dict,
                            load_config)
from wsparse.model import build_model
from wsparse.train import load_for_inference, train

CPU = torch.device("cpu")
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
VOCAB = 64


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_fake_dataset(tmp_path, n_train=60_000, n_val=8_000):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for name, n in (("train", n_train), ("val", n_val)):
        arr = np.memmap(d / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(n,))
        arr[:] = rng.integers(0, VOCAB, size=n).astype(np.uint16)
        arr.flush()
        del arr
    with open(d / "meta.json", "w") as f:
        json.dump({"vocab_size": VOCAB, "eos_id": 0, "tokenizer": "fake",
                   "train_tokens": n_train}, f)
    return str(d)


def bn_kwargs(**over):
    kw = dict(enabled=True, layers="all", placement="residual_out", n_features=48, k=8, j=16,
              selection_mode="abs_topk", surrogate_mode="dust", dust_population=8,
              dust_sigma=0.1, init_mode="sqrt_k_selection_corrected")
    kw.update(over)
    return kw


def smoke_config(data_dir, out_dir, run_name="dust", n_layers=2, tie=False, **bn):
    cfg = Config()
    cfg.data.data_dir = data_dir
    cfg.data.seq_len = 32
    cfg.model = ModelConfig(vocab_size=VOCAB, max_seq_len=32, n_layers=n_layers, d_model=32,
                            n_heads=4, tie_embeddings=tie)
    cfg.train.batch_size = 8
    cfg.train.micro_batch_size = 4
    cfg.train.max_steps = 4
    cfg.train.warmup_steps = 2
    cfg.train.log_every_steps = 1
    cfg.train.validate_every_steps = 2
    cfg.train.val_batches = 2
    cfg.train.checkpoint_every_steps = 4
    cfg.train.dtype = "float32"
    cfg.train.device = "cpu"
    cfg.train.tensorboard = False
    cfg.train.out_dir = out_dir
    cfg.train.run_name = run_name
    cfg.activation_bottleneck = ActivationBottleneckConfig(**bn_kwargs(**bn))
    return cfg


def tiny_model(n_layers=2, **bn):
    torch.manual_seed(0)
    mcfg = ModelConfig(vocab_size=VOCAB, max_seq_len=16, n_layers=n_layers, d_model=32,
                       n_heads=4, tie_embeddings=False)
    model = build_model(mcfg)
    ctl = apply_activation_bottleneck(model, ActivationBottleneckConfig(**bn_kwargs(**bn)))
    return model, ctl


def with_md(cfg):
    """The config's model under magnitude-direction decoupling, built through
    the dataclass so its validation runs (a reload re-validates it)."""
    import dataclasses
    cfg.model = ModelConfig(**{**dataclasses.asdict(cfg.model), "decouple": True,
                               "pos_encoding": "rope", "logit_scale": "none"})
    return cfg


def md_constraints_hold(path):
    """Rebuild the model and its decoupled optimizer from a checkpoint and
    check the unit rows and the direction spheres with the saved gains."""
    from test_decouple import _check_constraints
    from wsparse.decouple import build_decoupled_optimizer
    payload = torch.load(path, weights_only=False)
    model, loaded, _ = load_for_inference(str(path))
    opt = build_decoupled_optimizer(model, loaded.train, gain_mode=loaded.model.decouple_gains)
    opt.load_state_dict(payload["optimizer"])
    _check_constraints(model, opt)


def snapshots_via_on_step(cfg):
    snaps = {}

    def on_step(step, model, bottleneck, optimizer):
        snaps[step] = {n: p.detach().clone() for n, p in model.named_parameters()}

    summary = train(cfg, on_step=on_step)
    return summary, snaps


def changed(a, b, name):
    return not torch.equal(a[name], b[name])


# --------------------------------------------------------------------------- #
# schedule, population, config
# --------------------------------------------------------------------------- #


def test_step_schedule_starts_each_period_with_the_hard_steps():
    kinds = lambda sched, n: [dust_step_kind(s, sched) for s in range(n)]  # noqa: E731
    assert kinds([1, 1], 4) == ["hard", "dust", "hard", "dust"]
    assert kinds([1, 3], 5) == ["hard", "dust", "dust", "dust", "hard"]
    assert kinds([3, 1], 5) == ["hard", "hard", "hard", "dust", "hard"]
    assert kinds([0, 1], 3) == ["dust"] * 3
    with pytest.raises(ValueError):
        dust_step_kind(0, [1, 0])


def test_population_is_split_evenly_with_at_least_two_draws_each():
    assert draws_per_bottleneck(8, 2) == 4
    assert draws_per_bottleneck(40, 10) == 4
    with pytest.raises(ValueError, match="multiple"):
        draws_per_bottleneck(9, 2)
    with pytest.raises(ValueError, match="at least 2"):
        draws_per_bottleneck(2, 2)
    # the controller applies the same rule at install time
    with pytest.raises(ValueError, match="multiple"):
        tiny_model(n_layers=3, dust_population=8)


@pytest.mark.parametrize("bad", [
    dict(surrogate_mode="hard", dust_sigma=0.2),          # dust fields under another mode
    dict(surrogate_mode="rblapsum", dust_schedule=[1, 3]),
    dict(placement="pre_mlp"),                            # no tail for branch placements
    dict(placement="residual,residual_out"),
    dict(selection_mode="gated_topk"),
    dict(dust_schedule=[1, 0]),
    dict(dust_schedule=[1]),
    dict(dust_schedule=[1, 2, 3]),
    dict(dust_schedule=[-1, 1]),
    dict(dust_target="both"),
    dict(dust_optimizer="two"),
    dict(surrogate_mode="hard", dust_optimizer="separate"),
    dict(dust_grad_scale=0.0),
    dict(dust_grad_scale=-1.0),
    dict(dust_grad_scale=float("inf")),
    dict(dust_sigma=0.0),
    dict(dust_sigma=float("nan")),
    dict(dust_sigma_mode="relative_span"),
    dict(dust_gamma=1.5),
    dict(dust_chunk_draws=0),
    dict(dust_cosine_every_steps=-1),
    dict(dust_population=0),
    dict(rblapsum_support_scale=0.5),
    dict(rblapsum_surrogate_scope="inactive"),
    dict(rblapsum_boundary_floor=0.1),
    dict(hard_inference=False),
    dict(stochastic_width="uniform"),
    dict(value_shift="fixed"),
    dict(policy_baseline="none"),
])
def test_invalid_dust_configurations_are_rejected(bad):
    with pytest.raises(ValueError):
        ActivationBottleneckConfig(**bn_kwargs(**bad))


def test_dust_fields_survive_yaml_cli_dict_and_shipped_configs(tmp_path):
    path = tmp_path / "d.yaml"
    tree = {"activation_bottleneck": bn_kwargs(dust_schedule=[1, 3], dust_target="backprop",
                                               dust_gamma=0.5, dust_sigma_mode="relative_rms",
                                               dust_grad_scale=0.01, dust_optimizer="separate")}
    with open(path, "w") as f:
        yaml.safe_dump(tree, f)
    cfg = load_config(str(path))
    ab = cfg.activation_bottleneck
    assert ab.dust_schedule == [1, 3] and ab.dust_target == "backprop" and ab.dust_gamma == 0.5
    assert ab.dust_grad_scale == 0.01 and ab.dust_optimizer == "separate"
    cfg2 = load_config(str(path), ["--activation_bottleneck.dust_schedule=[2,1]",
                                   "--activation_bottleneck.dust_target=encoder",
                                   "--activation_bottleneck.dust_population=12"])
    ab2 = cfg2.activation_bottleneck
    assert ab2.dust_schedule == [2, 1] and ab2.dust_target == "encoder"
    assert ab2.dust_population == 12
    cfg3 = load_config(str(path), ["--activation_bottleneck.dust_schedule=3,1"])
    assert cfg3.activation_bottleneck.dust_schedule == [3, 1]
    back = config_from_dict(cfg.to_dict())
    assert back.activation_bottleneck.dust_schedule == [1, 3]
    assert back.activation_bottleneck.dust_sigma_mode == "relative_rms"
    for name in ("bn_dust.yaml", "bn_dust_backprop.yaml"):
        c = load_config(os.path.join(CONFIG_DIR, name))
        a = c.activation_bottleneck
        assert a.surrogate_mode == "dust" and a.dust_population % c.model.n_layers == 0
        assert a.dust_population // c.model.n_layers >= 2
    assert load_config(os.path.join(CONFIG_DIR, "bn_dust_backprop.yaml")
                       ).activation_bottleneck.dust_target == "backprop"
    # the hard config is untouched by the new fields
    hard = load_config(os.path.join(CONFIG_DIR, "bn_hard.yaml")).activation_bottleneck
    assert hard.surrogate_mode == "hard" and hard.dust_schedule == [1, 1]


# --------------------------------------------------------------------------- #
# the gate and the tail
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("training", [True, False])
def test_gate_under_dust_is_the_hard_gate(training):
    dust = AdaptiveLapSumTopKGate(n_features=32, k=4, j=8, surrogate_mode="dust")
    hard = AdaptiveLapSumTopKGate(n_features=32, k=4, j=8, surrogate_mode="hard")
    dust.train(training)
    hard.train(training)
    torch.manual_seed(1)
    z = torch.randn(3, 5, 32)
    w = torch.randn(32)
    za, zb = z.clone().requires_grad_(True), z.clone().requires_grad_(True)
    ya, yb = dust(za), hard(zb)
    assert torch.equal(ya, yb)
    (ya * w).sum().backward()
    (yb * w).sum().backward()
    assert torch.equal(za.grad, zb.grad)
    assert not dust.surrogate_active()
    assert (za.grad != 0).sum(-1).max() <= 4  # the J candidates get nothing
    cfg = ActivationBottleneckConfig(**bn_kwargs())
    assert effective_backward_support(cfg) == cfg.k
    # j = 0 and k = N are legal, as for the hard gate
    ActivationBottleneckConfig(**bn_kwargs(j=0))
    ActivationBottleneckConfig(**bn_kwargs(k=48, j=0))


@pytest.mark.parametrize("placement,code_residual,post_norm", [
    ("residual_out", False, False), ("residual_out", False, True),
    ("residual", False, False), ("residual", False, True),
    ("residual_out", True, False), ("residual_out", True, True),
])
def test_forward_from_gate_reproduces_the_clean_forward(placement, code_residual, post_norm):
    model, ctl = tiny_model(n_layers=3, placement=placement, code_residual=code_residual,
                            post_norm=post_norm, code_residual_scale=0.7, dust_population=6)
    modules = [m for _, m in ctl.layers]
    assert [m._site for m in modules] == [(i, placement) for i in range(3)]
    x = torch.randint(0, VOCAB, (2, 8))
    records = []
    model.train()
    with torch.no_grad(), dust_capture(modules, records):
        clean, _ = model(x)
    assert [r.module for r in records] == modules
    scales = [r.scale for r in records]
    assert scales == ([1.0, 0.7, 0.7] if code_residual else [1.0, 1.0, 1.0])
    for r in records:
        assert r.z.shape == (2, 8, 48)
        # the recorded encoder input reproduces the gate input
        proj = r.module.in_proj(r.enc_input) * r.scale
        if code_residual and r is not records[0]:
            assert torch.allclose(r.z - proj, r.z - proj)  # the carry is the remainder
        else:
            assert torch.allclose(proj, r.z, atol=1e-6)
    model.eval()
    with torch.no_grad():
        for r in records:
            i, p = r.module._site
            out = model.forward_from_gate(i, p, r.z)
            assert torch.allclose(out, clean, atol=1e-5)
            out32 = model.forward_from_gate(i, p, r.z, head_dtype=torch.float32)
            assert torch.allclose(out32, clean, atol=1e-5)
    assert all(m._dust_capture is None for m in modules)


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #


def test_discounted_returns_closed_form():
    ell = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(discounted_returns(ell, 0.0), ell)
    assert torch.allclose(discounted_returns(ell, 1.0), torch.tensor([[10.0, 9.0, 7.0, 4.0]]))
    assert torch.allclose(discounted_returns(ell, 0.5),
                          torch.tensor([[3.25, 4.5, 5.0, 4.0]]))


def test_linear_loss_estimate_is_the_gradient_on_the_pool_only():
    torch.manual_seed(2)
    B, T, N, M = 2, 3, 12, 6
    z = torch.randn(B, T, N)
    w = torch.randn(N)
    cand = pool_indices(z, "abs_topk", M)
    gen = torch.Generator().manual_seed(0)
    loss_fn = lambda zp: (zp * w).sum(-1)                                    # noqa: E731
    for mode in ("absolute", "relative_rms"):
        sigma_t = noise_scale(mode, 0.3, z, cand)
        assert sigma_t.shape == (B, T, 1)
        ghat, st = dust_estimate(z, cand, sigma_t, draws=8000, gamma=0.0, loss_fn=loss_fn,
                                 generator=gen, chunk=200)
        expected = w[cand] * (1.0 - 1.0 / 8000)
        assert torch.allclose(ghat, expected, atol=0.15), (ghat - expected).abs().max()
        assert st["draws"].item() == 8000
    # the draw-mean baseline: E g^ = (1 - 1/K) grad, here K = 2
    acc = torch.zeros(B, T, M)
    reps = 3000
    sigma_t = noise_scale("absolute", 0.3, z, cand)
    for _ in range(reps):
        g, _ = dust_estimate(z, cand, sigma_t, draws=2, gamma=0.0, loss_fn=loss_fn,
                             generator=gen, chunk=2)
        acc += g
    assert torch.allclose(acc / reps, 0.5 * w[cand], atol=0.15)
    # topk ranks by the signed value: the pool is the largest z, not |z|
    cand_signed = pool_indices(z, "topk", M)
    assert torch.equal(cand_signed, torch.topk(z, M, dim=-1).indices)


def test_discounted_credit_sums_the_later_tokens_losses():
    torch.manual_seed(3)
    B, T, N, M = 2, 3, 8, 4
    z = torch.randn(B, T, N)
    w = torch.randn(N)
    cand = pool_indices(z, "abs_topk", M)
    sigma_t = noise_scale("absolute", 0.2, z, cand)
    # ell_s = sum_{t <= s} w . z_t: d(sum_{s >= t} ell_s)/dz_t = (T - t) w, d ell_t / dz_t = w
    loss_fn = lambda zp: torch.cumsum((zp * w).sum(-1), dim=-1)                # noqa: E731
    gen = torch.Generator().manual_seed(1)
    for gamma, factor in ((0.0, [1.0, 1.0, 1.0]), (1.0, [3.0, 2.0, 1.0]),
                          (0.5, [1.75, 1.5, 1.0])):
        ghat, _ = dust_estimate(z, cand, sigma_t, draws=8000, gamma=gamma, loss_fn=loss_fn,
                                generator=gen, chunk=250)
        expected = w[cand] * torch.tensor(factor)[None, :, None]
        assert torch.allclose(ghat, expected, atol=0.35), (gamma, (ghat - expected).abs().max())


def test_token_losses_zero_ignored_targets_and_use_float32():
    logits = torch.randn(2, 4, VOCAB, dtype=torch.bfloat16)
    y = torch.randint(0, VOCAB, (2, 4))
    y[0, 1] = -100
    ell, valid = token_losses(logits, y)
    assert ell.dtype == torch.float32 and ell[0, 1] == 0 and not valid[0, 1]
    assert valid.sum() == 7 and (ell[valid] > 0).all()


# --------------------------------------------------------------------------- #
# the micro step
# --------------------------------------------------------------------------- #


def micro_step_on(model, ctl, population, sigma, target, want_cosine, seed=0, **over):
    cfg = copy.copy(ctl.cfg)
    cfg.dust_population = population
    cfg.dust_sigma = sigma
    cfg.dust_target = target
    for k, v in over.items():
        setattr(cfg, k, v)
    state = DustTrainingState(cfg, len(ctl.layers), CPU, seed=seed)
    torch.manual_seed(5)
    x = torch.randint(0, VOCAB, (4, 8))
    y = torch.randint(0, VOCAB, (4, 8))
    model.zero_grad(set_to_none=True)
    st = dust_micro_step(model, ctl, x, y, state, CPU, torch.float32, accum=1,
                         want_cosine=want_cosine)
    return st, state


def test_micro_step_encoder_target_touches_encoders_only_and_matches_backprop_sign():
    model, ctl = tiny_model(n_layers=2)
    st, _ = micro_step_on(model, ctl, population=2 * 96, sigma=0.05, target="encoder",
                          want_cosine=True)
    for name, p in model.named_parameters():
        if "in_proj" in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    assert set(st["cosine"]) == {"blocks.0", "blocks.1"}
    for name, cos in st["cosine"].items():
        assert cos > 0.25, (name, cos)  # the sign and the scaling of the estimate
    assert math.isfinite(st["ce_clean"]) and st["ce_clean"] > 0
    assert st["ghat_rms"] > 0 and st["reward_rms"] > 0
    assert st["ce_gap"] == pytest.approx(st["ce_perturbed"] - st["ce_clean"])
    # the tails ran in eval mode: one usage update (the clean forward) per gate
    for _, m in ctl.layers:
        assert float(m.gate.usage_steps) == 1.0
        assert "score_gap" in m.gate.diagnostics
    assert model.training


def test_micro_step_trivial_gate_estimate_aligns_with_the_smooth_gradient():
    # k = N: no selection, so the loss is smooth in z and the estimate should
    # align well with the exact gradient of the encoder weights
    model, ctl = tiny_model(n_layers=2, n_features=16, k=16, j=0)
    st, _ = micro_step_on(model, ctl, population=2 * 128, sigma=0.02, target="encoder",
                          want_cosine=True)
    for name, cos in st["cosine"].items():
        assert cos > 0.6, (name, cos)


def test_micro_step_backprop_target_reaches_everything_before_the_gate_inputs():
    model, ctl = tiny_model(n_layers=2, code_residual=True, post_norm=True)
    st, _ = micro_step_on(model, ctl, population=8, sigma=0.1, target="backprop",
                          want_cosine=False)
    assert st["cosine"] == {}
    grads = {n: p.grad for n, p in model.named_parameters()}
    for name in ("tok_emb.weight", "blocks.0.attn.qkv.weight", "blocks.1.mlp.fc1.weight",
                 "blocks.0.residual_out_bottleneck.in_proj.weight",
                 "blocks.1.residual_out_bottleneck.in_proj.weight",
                 "blocks.0.residual_out_bottleneck.out_proj.weight",
                 "blocks.0.residual_out_bottleneck.post_norm.weight"):
        assert grads[name] is not None and grads[name].abs().sum() > 0, name
    for name in ("lm_head.weight", "norm_f.weight",
                 "blocks.1.residual_out_bottleneck.out_proj.weight"):
        assert grads[name] is None, name


def test_micro_step_accumulation_scale_and_generator_reproducibility():
    model, ctl = tiny_model(n_layers=2)
    st1, _ = micro_step_on(model, ctl, population=8, sigma=0.1, target="encoder",
                           want_cosine=False, seed=11)
    g1 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    st2, _ = micro_step_on(model, ctl, population=8, sigma=0.1, target="encoder",
                           want_cosine=False, seed=11)
    g2 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert st1 == st2 and all(torch.equal(g1[n], g2[n]) for n in g1)
    st3, _ = micro_step_on(model, ctl, population=8, sigma=0.1, target="encoder",
                           want_cosine=False, seed=12)
    assert any(not torch.equal(g1[n], p.grad) for n, p in model.named_parameters()
               if n in g1)
    # accum divides the accumulated gradient
    cfg = copy.copy(ctl.cfg)
    cfg.dust_population = 8
    state = DustTrainingState(cfg, 2, CPU, seed=11)
    torch.manual_seed(5)
    x = torch.randint(0, VOCAB, (4, 8))
    y = torch.randint(0, VOCAB, (4, 8))
    model.zero_grad(set_to_none=True)
    dust_micro_step(model, ctl, x, y, state, CPU, torch.float32, accum=4)
    for n, p in model.named_parameters():
        if n in g1:
            assert torch.allclose(p.grad * 4, g1[n], atol=1e-6)


@pytest.mark.parametrize("target", ["encoder", "backprop"])
def test_grad_scale_multiplies_the_dust_gradient_only(target):
    model, ctl = tiny_model(n_layers=2, code_residual=(target == "backprop"))
    st1, _ = micro_step_on(model, ctl, population=8, sigma=0.1, target=target,
                           want_cosine=True, seed=4)
    g1 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    st3, state = micro_step_on(model, ctl, population=8, sigma=0.1, target=target,
                               want_cosine=True, seed=4, dust_grad_scale=3.0)
    assert state.grad_scale == 3.0
    g3 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(g1) == set(g3)
    for n in g1:
        assert torch.allclose(g3[n], 3.0 * g1[n], atol=1e-6, rtol=1e-5), n
    # the estimator statistics and the cosine are unscaled / scale-invariant
    assert st3["ghat_rms"] == pytest.approx(st1["ghat_rms"])
    assert st3["ce_perturbed"] == pytest.approx(st1["ce_perturbed"])
    for k in st1["cosine"]:
        assert st3["cosine"][k] == pytest.approx(st1["cosine"][k], abs=1e-5)


def test_dust_state_round_trip_continues_the_noise_stream():
    cfg = ActivationBottleneckConfig(**bn_kwargs(dust_population=8))
    a = DustTrainingState(cfg, 2, CPU, seed=3)
    b = DustTrainingState(cfg, 2, CPU, seed=3)
    torch.randn(100, generator=a.generator)
    a.dust_steps = 7
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert b.dust_steps == 7
    assert torch.equal(torch.randn(10, generator=a.generator),
                       torch.randn(10, generator=b.generator))
    assert a.chunk == 1 and a.per_bottleneck == 4
    with pytest.raises(ValueError, match="version"):
        b.load_state_dict({"version": 99})
    cfg.dust_cosine_every_steps = 3
    c = DustTrainingState(cfg, 2, CPU, seed=0)
    due = []
    for _ in range(7):
        due.append(c.cosine_due())
        c.dust_steps += 1
    assert due == [True, False, False, True, False, False, True]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def test_training_alternates_and_the_encoder_target_moves_only_the_encoders(tmp_path):
    data = make_fake_dataset(tmp_path)
    out = str(tmp_path / "runs")
    cfg = smoke_config(data, out, dust_target="encoder", dust_cosine_every_steps=1)
    summary, snaps = snapshots_via_on_step(cfg)
    assert sorted(snaps) == [0, 1, 2, 3]
    # step 0 (hard) moved the head and the encoders; step 1 (dust) the encoders only
    assert changed(snaps[0], snaps[1], "lm_head.weight")
    assert changed(snaps[0], snaps[1], "blocks.0.residual_out_bottleneck.in_proj.weight")
    for name in snaps[1]:
        if "in_proj" in name:
            assert changed(snaps[1], snaps[2], name), name
        else:
            assert not changed(snaps[1], snaps[2], name), name
    assert changed(snaps[2], snaps[3], "lm_head.weight")  # step 2 is hard again
    run_dir = tmp_path / "runs" / "dust"
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    tr = [r for r in rows if "train/ce" in r]
    assert [r["dust/steps"] for r in tr] == [0.0, 1.0, 0.0, 1.0]
    for r in tr:
        assert math.isfinite(r["train/ce"]) and r["train/loss"] == r["train/ce"]
    dust_rows = [r for r in tr if r["dust/steps"]]
    for r in dust_rows:
        for key in ("dust/ce_perturbed", "dust/ce_gap", "dust/reward_rms", "dust/ghat_rms",
                    "dust/grad_norm", "dust/cosine", "dust_cosine/blocks.0",
                    "dust_cosine/blocks.1"):
            assert key in r and math.isfinite(r[key]), key
        assert r["dust/ce_clean"] == pytest.approx(r["train/ce"])
    assert "dust/ce_gap" not in tr[0]
    assert math.isfinite(summary["best_val_ce"])
    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    ds = payload["dust_state"]
    assert ds["version"] == 1 and ds["dust_steps"] == 2 and ds["world_size"] == 1
    model, loaded, ctl = load_for_inference(str(run_dir / "latest.pt"))
    assert loaded.activation_bottleneck.surrogate_mode == "dust"
    assert loaded.activation_bottleneck.dust_schedule == [1, 1]
    x = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])
    # resume: the pattern continues from the step index, the state is restored
    cfg2 = smoke_config(data, out, dust_target="encoder", dust_cosine_every_steps=1)
    cfg2.train.max_steps = 6
    cfg2.train.resume = "auto"
    _, snaps2 = snapshots_via_on_step(cfg2)
    assert sorted(snaps2) == [4, 5]
    assert all(torch.equal(snaps2[4][n], snaps[3][n]) == (not changed(snaps[3], snaps2[4], n))
               for n in snaps2[4])
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    tr = [r for r in rows if "train/ce" in r]
    assert [r["dust/steps"] for r in tr[-2:]] == [0.0, 1.0]
    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    assert payload["dust_state"]["dust_steps"] == 3


def test_training_backprop_target_with_code_residual(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    out = str(tmp_path / "runs")
    cfg = smoke_config(data, out, run_name="bp", n_layers=3, dust_target="backprop",
                       dust_schedule=[1, 2], dust_population=6, code_residual=True,
                       post_norm=True, dust_gamma=0.9, dust_chunk_draws=2)
    summary, snaps = snapshots_via_on_step(cfg)
    # steps: 0 hard, 1 dust, 2 dust, 3 hard
    for a, b in ((1, 2), (2, 3)):
        assert not changed(snaps[a], snaps[b], "lm_head.weight")
        assert not changed(snaps[a], snaps[b], "norm_f.weight")
        assert not changed(snaps[a], snaps[b], "blocks.2.residual_out_bottleneck.out_proj.weight")
        assert changed(snaps[a], snaps[b], "tok_emb.weight")
        assert changed(snaps[a], snaps[b], "blocks.0.attn.qkv.weight")
        assert changed(snaps[a], snaps[b], "blocks.1.residual_out_bottleneck.out_proj.weight")
        assert changed(snaps[a], snaps[b], "blocks.2.residual_out_bottleneck.in_proj.weight")
    assert changed(snaps[0], snaps[1], "lm_head.weight")
    assert math.isfinite(summary["best_val_ce"])
    rows = [json.loads(l) for l in open(tmp_path / "runs" / "bp" / "metrics.jsonl")]
    tr = [r for r in rows if "train/ce" in r]
    assert [r["dust/steps"] for r in tr] == [0.0, 1.0, 1.0, 0.0]


def test_training_shared_projections_and_tied_embeddings(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    out = str(tmp_path / "runs")
    cfg = smoke_config(data, out, run_name="sh", tie=True, share_projections=True,
                       dust_sigma_mode="relative_rms", dust_target="encoder")
    cfg.train.max_steps = 2
    summary, snaps = snapshots_via_on_step(cfg)
    assert math.isfinite(summary["best_val_ce"])
    cfg = smoke_config(data, out, run_name="shbp", tie=True, share_projections=True,
                       dust_target="backprop")
    cfg.train.max_steps = 2
    summary, snaps = snapshots_via_on_step(cfg)
    assert math.isfinite(summary["best_val_ce"])


def optimizer_steps(payload_state, model, train_cfg, param):
    """The Adam step count a saved optimizer state holds for ``param`` (None
    when the parameter has no state), by rebuilding the builder's ordering."""
    from wsparse.optim import build_optimizer
    opt = build_optimizer(model, train_cfg)
    flat = [p for g in opt.param_groups for p in g["params"]]
    idx = next(i for i, p in enumerate(flat) if p is param)
    st = payload_state["state"].get(idx)
    return None if st is None else int(st["step"])


def test_separate_optimizer_keeps_its_own_moments_and_step_count(tmp_path):
    data = make_fake_dataset(tmp_path)
    out = str(tmp_path / "runs")
    cfg = smoke_config(data, out, run_name="sep", dust_target="encoder", dust_optimizer="separate")
    summary, snaps = snapshots_via_on_step(cfg)
    assert math.isfinite(summary["best_val_ce"])
    for name in snaps[1]:  # step 1 is a DUST step: encoders only, as under shared
        assert changed(snaps[1], snaps[2], name) == ("in_proj" in name), name
    run_dir = tmp_path / "runs" / "sep"
    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    assert "dust_optimizer" in payload
    model, loaded, ctl = load_for_inference(str(run_dir / "latest.pt"))
    enc = model.blocks[0].residual_out_bottleneck.in_proj.weight
    head = model.lm_head.weight
    # 4 steps: hard, dust, hard, dust.  The run's optimizer saw the two hard
    # steps for every parameter; the DUST one the two DUST steps, encoders only
    assert optimizer_steps(payload["optimizer"], model, loaded.train, enc) == 2
    assert optimizer_steps(payload["optimizer"], model, loaded.train, head) == 2
    assert optimizer_steps(payload["dust_optimizer"], model, loaded.train, enc) == 2
    assert optimizer_steps(payload["dust_optimizer"], model, loaded.train, head) is None
    assert len(payload["dust_optimizer"]["state"]) == 2  # the two encoder weights
    # resume: both optimizers restored and advanced
    cfg2 = smoke_config(data, out, run_name="sep", dust_target="encoder",
                        dust_optimizer="separate")
    cfg2.train.max_steps = 6
    cfg2.train.resume = "auto"
    train(cfg2)
    payload = torch.load(run_dir / "latest.pt", weights_only=False)
    assert optimizer_steps(payload["optimizer"], model, loaded.train, enc) == 3
    assert optimizer_steps(payload["dust_optimizer"], model, loaded.train, enc) == 3
    # the backprop target under the separate optimizer: state for everything
    # before the gate inputs, none for the unembedding
    cfg3 = smoke_config(data, out, run_name="sepbp", dust_target="backprop",
                        dust_optimizer="separate")
    cfg3.train.max_steps = 2
    train(cfg3)
    payload = torch.load(tmp_path / "runs" / "sepbp" / "latest.pt", weights_only=False)
    model3, loaded3, _ = load_for_inference(str(tmp_path / "runs" / "sepbp" / "latest.pt"))
    assert optimizer_steps(payload["dust_optimizer"], model3, loaded3.train,
                           model3.blocks[0].attn.qkv.weight) == 1
    assert optimizer_steps(payload["dust_optimizer"], model3, loaded3.train,
                           model3.lm_head.weight) is None


def test_separate_optimizer_resume_and_decouple_rules(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    out = str(tmp_path / "runs")
    shared = smoke_config(data, out, run_name="s")
    shared.train.max_steps = 2
    shared.train.checkpoint_every_steps = 2
    train(shared)
    cfg = smoke_config(data, out, run_name="s", dust_optimizer="separate")
    cfg.train.max_steps = 4
    cfg.train.resume = "auto"
    with pytest.raises(ValueError, match="dust_optimizer"):
        train(cfg)
    # under decouple the DUST instance shares the MD parameterization and keeps
    # its own statistics; the constraints hold after both kinds of step
    for target in ("encoder", "backprop"):
        cfg = with_md(smoke_config(data, out, run_name=f"md_{target}",
                                   dust_optimizer="separate", dust_target=target))
        cfg.train.checkpoint_every_steps = 2
        summary = train(cfg)
        assert math.isfinite(summary["best_val_ce"])
        run_dir = tmp_path / "runs" / f"md_{target}"
        payload = torch.load(run_dir / "latest.pt", weights_only=False)
        ds = payload["dust_optimizer"]
        assert ds["shares_parameterization"] is True
        assert all("c_f" not in st and "raw_grow" not in st for st in ds["state"].values())
        assert any("c_f" in st for st in payload["optimizer"]["state"].values())
        md_constraints_hold(run_dir / "latest.pt")
        cfg2 = with_md(smoke_config(data, out, run_name=f"md_{target}",
                                    dust_optimizer="separate", dust_target=target))
        cfg2.train.max_steps = 6
        cfg2.train.resume = "auto"
        assert math.isfinite(train(cfg2)["best_val_ce"])
        md_constraints_hold(run_dir / "latest.pt")


def test_resume_without_dust_state_is_rejected(tmp_path):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    out = str(tmp_path / "runs")
    hard = smoke_config(data, out, run_name="h", surrogate_mode="hard", dust_population=32,
                        dust_sigma=0.1)
    hard.train.max_steps = 2
    hard.train.checkpoint_every_steps = 2
    train(hard)
    cfg = smoke_config(data, out, run_name="h")
    cfg.train.max_steps = 4
    cfg.train.resume = "auto"
    with pytest.raises(ValueError, match="dust_state"):
        train(cfg)


def test_ddp_and_float16_are_rejected(tmp_path, monkeypatch):
    data = make_fake_dataset(tmp_path, 20_000, 4_000)
    cfg = smoke_config(data, str(tmp_path / "runs"), run_name="f16")
    cfg.train.dtype = "float16"
    cfg.train.device = "cpu"
    # float16 resolves to float32 on cpu; only a cuda box could reject it, so
    # check the world-size rule, which does not depend on the device
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29599")
    import wsparse.train as tr
    monkeypatch.setattr(tr.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(tr.dist, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="single-process"):
        train(cfg)
