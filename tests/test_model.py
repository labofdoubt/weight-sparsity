import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsparse.config import ModelConfig
from wsparse.model import MLP, RMSNorm, build_model


def tiny_cfg(**kw):
    base = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, mlp_ratio=4.0)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_and_loss():
    torch.manual_seed(0)
    model = build_model(tiny_cfg())
    x = torch.randint(0, 97, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 97)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # Untrained loss should be close to ln(vocab) -- but only on the *next*-token
    # objective.  Passing x as its own labels scores predicting token t at
    # position t, which a tied head does for free by reading the token embedding
    # back out of the residual stream, so that number is far below ln(vocab) and
    # says nothing about the init.
    shifted = F.cross_entropy(
        logits[:, :-1].reshape(-1, 97).float(), x[:, 1:].reshape(-1)
    )
    assert abs(shifted.item() - math.log(97)) < 1.0


def test_no_biases_by_default():
    model = build_model(tiny_cfg())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None, name
    assert all("bias" not in n for n, _ in model.named_parameters())


def test_biases_can_be_enabled():
    model = build_model(tiny_cfg(bias=True))
    assert model.blocks[0].mlp.fc1.bias is not None
    assert model.lm_head.bias is None  # the head stays bias-free


def test_rmsnorm_matches_reference():
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    x = torch.randn(4, 8)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(norm(x), ref, atol=1e-5)
    assert isinstance(build_model(tiny_cfg()).norm_f, RMSNorm)


def test_learnable_positional_embeddings():
    model = build_model(tiny_cfg())
    assert isinstance(model.pos_emb, nn.Embedding)
    assert model.pos_emb.weight.requires_grad
    assert model.pos_emb.weight.shape == (32, 32)


def test_weight_tying():
    tied = build_model(tiny_cfg(tie_embeddings=True))
    assert tied.lm_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()
    untied = build_model(tiny_cfg(tie_embeddings=False))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert untied.num_parameters() > tied.num_parameters()


def test_mlp_ratio_changes_hidden_width():
    m2 = build_model(tiny_cfg(mlp_ratio=2.0))
    m8 = build_model(tiny_cfg(mlp_ratio=8.0))
    assert m2.blocks[0].mlp.fc1.out_features == 64
    assert m8.blocks[0].mlp.fc1.out_features == 256


def test_swiglu_keeps_parameter_budget():
    cfg = tiny_cfg(mlp_activation="swiglu")
    mlp = MLP(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    assert mlp(x).shape == (2, 5, cfg.d_model)
    assert mlp.fc1.out_features == 2 * cfg.d_mlp


def test_init_std_is_respected():
    model = build_model(tiny_cfg(init_std=0.05, init_scale_residual=False, n_layers=4, d_model=128))
    std = model.blocks[0].mlp.fc1.weight.std().item()
    assert 0.04 < std < 0.06


def test_embedding_init_defaults_to_unit_variance_sum():
    torch.manual_seed(0)
    cfg = tiny_cfg(d_model=256, vocab_size=2048, max_seq_len=256, tie_embeddings=True)
    model = build_model(cfg)
    expected = 1.0 / math.sqrt(2.0)
    for emb in (model.tok_emb, model.pos_emb):
        assert abs(emb.weight.std().item() - expected) < 0.05 * expected
    # tok_emb + pos_emb is ~unit variance per element (averaged over the table,
    # so the sample variance is tight enough to assert on)
    total = (model.tok_emb.weight[: cfg.max_seq_len] + model.pos_emb.weight).var().item()
    assert abs(total - 1.0) < 0.05


def test_embedding_std_overrides():
    model = build_model(tiny_cfg(d_model=256, init_std_embedding=0.05, init_std_pos=0.2))
    assert abs(model.tok_emb.weight.std().item() - 0.05) < 0.005
    assert abs(model.pos_emb.weight.std().item() - 0.2) < 0.02
    # init_std_pos falls back to init_std_embedding, not to the 1/sqrt(2) default
    model = build_model(tiny_cfg(d_model=256, init_std_embedding=0.05))
    assert abs(model.pos_emb.weight.std().item() - 0.05) < 0.005


def test_tied_lm_head_keeps_embedding_std():
    """lm_head must not overwrite the tied token embedding with the linear std."""
    model = build_model(
        tiny_cfg(d_model=256, tie_embeddings=True, init_scheme="fan_in", init_gain=1.0)
    )
    assert model.lm_head.weight is model.tok_emb.weight
    expected = 1.0 / math.sqrt(2.0)
    assert abs(model.tok_emb.weight.std().item() - expected) < 0.05 * expected


def test_unembedding_ignores_linear_init():
    """init_scheme / init_std / init_gain must not reach lm_head."""
    for kw in (
        dict(init_scheme="fan_in", init_gain=3.0),
        dict(init_scheme="fixed_std", init_std=0.5),
    ):
        model = build_model(tiny_cfg(d_model=256, tie_embeddings=False, **kw))
        expected = 1.0 / math.sqrt(256)  # the unembedding default, not the linear std
        assert abs(model.lm_head.weight.std().item() - expected) < 0.1 * expected, kw


def test_init_std_unembedding_sets_the_head_when_untied():
    model = build_model(tiny_cfg(d_model=256, tie_embeddings=False, init_std_unembedding=0.1))
    assert model.lm_head.weight is not model.tok_emb.weight
    assert abs(model.lm_head.weight.std().item() - 0.1) < 0.01
    # the embedding side is untouched by it
    assert abs(model.tok_emb.weight.std().item() - 1.0 / math.sqrt(2.0)) < 0.05


def test_init_std_unembedding_is_rejected_when_tied():
    with pytest.raises(ValueError, match="meaningless with tie_embeddings"):
        tiny_cfg(tie_embeddings=True, init_std_unembedding=0.1)


def test_tied_uses_one_std_for_both():
    model = build_model(tiny_cfg(d_model=256, tie_embeddings=True, init_std_embedding=0.1))
    assert model.lm_head.weight is model.tok_emb.weight
    assert abs(model.tok_emb.weight.std().item() - 0.1) < 0.01
    # pos_emb is separate and keeps the embedding default
    assert abs(model.pos_emb.weight.std().item() - 0.1) < 0.01


def test_auto_logit_scale_normalises_logits_to_unit_std():
    """alpha = 1 / (head_std * sqrt(d)) in every configuration."""
    torch.manual_seed(0)
    d, V = 256, 1024
    cases = [
        dict(tie_embeddings=True),                                    # head = 1/sqrt(2)
        dict(tie_embeddings=True, init_std_embedding=0.1),
        dict(tie_embeddings=False),                                   # head = 1/sqrt(d)
        dict(tie_embeddings=False, init_std_unembedding=1 / math.sqrt(2.0)),
    ]
    x = torch.randint(0, V, (2, 64))
    for kw in cases:
        model = build_model(tiny_cfg(d_model=d, vocab_size=V, max_seq_len=64, **kw)).eval()
        head_std = model.lm_head.weight.std().item()
        assert abs(model.logit_mult - 1.0 / (head_std * math.sqrt(d))) < 0.05, kw
        with torch.no_grad():
            logits, _ = model(x)
        assert 0.5 < logits.std().item() < 2.0, (kw, logits.std().item())


def test_untied_default_needs_no_rescaling():
    """1/sqrt(d) already puts the logits at unit std, so auto is a no-op there."""
    model = build_model(tiny_cfg(d_model=256, tie_embeddings=False))
    assert abs(model.logit_mult - 1.0) < 1e-9


def test_logit_scale_none_disables_it():
    assert build_model(tiny_cfg(logit_scale="none", tie_embeddings=True)).logit_mult == 1.0
    assert build_model(tiny_cfg(logit_scale="none", tie_embeddings=False)).logit_mult == 1.0


def test_fan_in_init():
    model = build_model(
        tiny_cfg(init_scheme="fan_in", init_gain=1.0, d_model=128, init_scale_residual=False)
    )
    w = model.blocks[0].mlp.fc1.weight
    expected = 1.0 / math.sqrt(w.shape[1])
    assert abs(w.std().item() - expected) < 0.2 * expected


def test_residual_scaling():
    cfg = tiny_cfg(n_layers=8, d_model=128, init_std=0.02)
    scaled = build_model(cfg)
    plain = build_model(tiny_cfg(n_layers=8, d_model=128, init_std=0.02, init_scale_residual=False))
    factor = 1 / math.sqrt(2 * 8)
    assert scaled.blocks[0].mlp.fc2.weight.std().item() < plain.blocks[0].mlp.fc2.weight.std().item()
    assert abs(scaled.blocks[0].attn.proj.weight.std().item() - 0.02 * factor) < 0.004


def test_causality():
    torch.manual_seed(0)
    model = build_model(tiny_cfg()).eval()
    x = torch.randint(0, 97, (1, 12))
    base, _ = model(x)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 97
    other, _ = model(x2)
    assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-5)


def test_generate_extends_sequence():
    model = build_model(tiny_cfg())
    out = model.generate(torch.randint(0, 97, (2, 4)), max_new_tokens=6, top_k=5)
    assert out.shape == (2, 10)
    assert out.max().item() < 97


def test_parameter_count_matches_hand_computation():
    cfg = ModelConfig(
        vocab_size=1000, max_seq_len=64, n_layers=3, d_model=64, n_heads=4, mlp_ratio=4.0
    )
    model = build_model(cfg)
    per_block = 4 * 64 * 64 + 2 * 64 * 256 + 2 * 64  # attn + mlp + two RMSNorm gains
    expected = 1000 * 64 + 64 * 64 + 3 * per_block + 64  # emb + pos + blocks + final norm
    assert model.num_parameters() == expected
