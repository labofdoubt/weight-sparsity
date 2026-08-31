import math

import torch
import torch.nn as nn

from wsparse.config import ModelConfig
from wsparse.model import MLP, RMSNorm, build_model


def tiny_cfg(**kw):
    base = dict(vocab_size=97, max_seq_len=32, n_layers=2, d_model=32, n_heads=4, mlp_ratio=4.0)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_and_loss():
    model = build_model(tiny_cfg())
    x = torch.randint(0, 97, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 97)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # untrained loss should be close to ln(vocab)
    assert abs(loss.item() - math.log(97)) < 1.0


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


def test_tied_logits_match_untied_scale():
    """logit_scale='auto' cancels the (large) tied embedding std in the logits."""
    kw = dict(d_model=256, vocab_size=1024, max_seq_len=64, n_layers=2)
    expected = 0.02 / (1.0 / math.sqrt(2.0))  # init_std / init_std_embedding
    torch.manual_seed(0)
    tied = build_model(tiny_cfg(tie_embeddings=True, **kw)).eval()
    assert abs(tied.logit_mult - expected) < 1e-6
    torch.manual_seed(0)
    untied = build_model(tiny_cfg(tie_embeddings=False, **kw)).eval()

    x = torch.randint(0, 1024, (2, 64))
    with torch.no_grad():
        tied_logits, _ = tied(x)
        untied_logits, _ = untied(x)
    # the tied head lands within 25% of the untied head's logit std ...
    ratio = tied_logits.std().item() / untied_logits.std().item()
    assert 0.75 < ratio < 1.25, ratio
    # ... and the real (shifted) LM loss starts near ln(vocab)
    with torch.no_grad():
        loss = nn.functional.cross_entropy(
            tied_logits[:, :-1].reshape(-1, 1024).float(), x[:, 1:].reshape(-1)
        )
    assert abs(loss.item() - math.log(1024)) < 1.0, loss.item()


def test_logit_scale_none_and_untied_are_unscaled():
    assert build_model(tiny_cfg(logit_scale="none", tie_embeddings=True)).logit_mult == 1.0
    # an untied head has its own linear-scale init, so "auto" leaves it alone
    assert build_model(tiny_cfg(tie_embeddings=False)).logit_mult == 1.0


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
