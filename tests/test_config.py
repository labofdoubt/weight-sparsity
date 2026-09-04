import copy
import os

import pytest

from wsparse.config import Config, SparsityConfig, load_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def test_defaults():
    cfg = Config()
    assert cfg.model.bias is False
    assert cfg.model.tie_embeddings is True
    assert cfg.train.optimizer == "adamw"
    assert cfg.sparsity.enabled is False
    assert cfg.train.grad_accum_steps == 1


def test_d_mlp_rounding():
    cfg = Config()
    cfg.model.d_model = 512
    assert Config().model.d_mlp == 3072  # 4 * 768
    from wsparse.config import ModelConfig

    assert ModelConfig(d_model=512, n_heads=8, mlp_ratio=2.5).d_mlp == 1280
    assert ModelConfig(d_model=100, n_heads=4, mlp_ratio=1.0).d_mlp % 8 == 0


@pytest.mark.parametrize(
    "name", ["dense.yaml", "ltp.yaml", "cs.yaml", "ltp_target.yaml", "cs_target.yaml", "smoke.yaml",
             "ltp_150m.yaml", "cs_150m.yaml"]
)
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(CONFIG_DIR, name))
    assert isinstance(cfg, Config)
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.max_seq_len >= cfg.data.seq_len


def test_base_composition_and_overrides():
    cfg = load_config(
        os.path.join(CONFIG_DIR, "ltp.yaml"),
        ["--train.lr=1e-3", "sparsity.beta_end=5e5", "--model.n_layers=2"],
    )
    assert cfg.train.lr == 1e-3
    assert cfg.sparsity.beta_end == 5e5
    assert cfg.model.n_layers == 2
    # values inherited from the included fragments survive
    assert cfg.sparsity.method == "ltp"
    assert cfg.model.d_model == 512


def test_nested_base_inherits():
    cfg = load_config(os.path.join(CONFIG_DIR, "ltp_target.yaml"))
    assert cfg.sparsity.method == "ltp"  # from sparsity/ltp.yaml via ltp_target.yaml
    assert cfg.sparsity.l0_coef == 0.0
    assert cfg.sparsity.target_density == 0.1
    assert cfg.sparsity.target_density_coef == 1.0


def test_type_coercion():
    cfg = load_config(None, ["train.compile=true", "train.betas=(0.9, 0.99)", "model.mlp_ratio=8"])
    assert cfg.train.compile is True
    assert cfg.train.betas == (0.9, 0.99)
    assert isinstance(cfg.model.mlp_ratio, float) and cfg.model.mlp_ratio == 8.0


def test_list_overrides_survive_shell_quote_stripping():
    # a shell turns --sparsity.targets=["mlp","attn"] into [mlp,attn]
    for text in ('["mlp","attn"]', "[mlp,attn]", "mlp,attn", "['mlp', 'attn']"):
        cfg = load_config(None, [f"sparsity.targets={text}"])
        assert cfg.sparsity.targets == ["mlp", "attn"]
    assert load_config(None, ["sparsity.targets=[mlp]"]).sparsity.targets == ["mlp"]


def test_scalar_string_override_is_not_split():
    cfg = load_config(None, ["train.run_name=my_run", "sparsity.beta_schedule=cosine"])
    assert cfg.train.run_name == "my_run"
    assert cfg.sparsity.beta_schedule == "cosine"


def test_bare_scalar_becomes_a_one_element_list():
    assert load_config(None, ["sparsity.targets=mlp"]).sparsity.targets == ["mlp"]
    with pytest.raises(ValueError, match="unknown sparsity targets"):
        load_config(None, ["sparsity.targets=mlpp"])


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(None, ["train.lr_typo=1e-3"])


def test_validation_errors():
    with pytest.raises(ValueError):
        SparsityConfig(method="nope")
    with pytest.raises(ValueError):
        SparsityConfig(targets=["mlp", "conv"])
    with pytest.raises(ValueError):
        SparsityConfig(target_density_coef=1.0)  # no target_density
    with pytest.raises(ValueError):
        SparsityConfig(target_density=1.5)


def test_cs_forces_grad_through_mask():
    cfg = SparsityConfig(method="cs", grad_through_mask=False)
    assert cfg.grad_through_mask is True  # no-op for CS, normalized away


def test_grad_accum():
    cfg = load_config(None, ["train.batch_size=64", "train.micro_batch_size=16"])
    assert cfg.train.grad_accum_steps == 4
    with pytest.raises(ValueError):
        load_config(None, ["train.batch_size=10", "train.micro_batch_size=4"])


def test_legacy_checkpoint_config_pins_logit_scale():
    """Pre-``logit_scale`` checkpoints must not pick up the "auto" default."""
    from wsparse.config import config_from_dict, load_config

    fresh = load_config().to_dict()
    assert fresh["model"]["logit_scale"] == "auto"  # yaml/default path is untouched

    legacy = copy.deepcopy(fresh)
    del legacy["model"]["logit_scale"]
    assert config_from_dict(legacy).model.logit_scale == "none"
    # a checkpoint that does carry the key keeps it
    assert config_from_dict(fresh).model.logit_scale == "auto"
