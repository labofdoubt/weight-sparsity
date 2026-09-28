import copy
import os

import pytest

from wsparse.config import Config, load_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def test_d_mlp_rounding():
    cfg = Config()
    cfg.model.d_model = 512
    assert Config().model.d_mlp == 3072  # 4 * 768
    from wsparse.config import ModelConfig

    assert ModelConfig(d_model=512, n_heads=8, mlp_ratio=2.5).d_mlp == 1280
    assert ModelConfig(d_model=100, n_heads=4, mlp_ratio=1.0).d_mlp % 8 == 0


@pytest.mark.parametrize(
    "name", ["bn_hard.yaml", "bn_lapsum.yaml", "bn_dense.yaml",
             "fineweb_rbk_500m.yaml"]
)
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(CONFIG_DIR, name))
    assert isinstance(cfg, Config)
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.model.max_seq_len >= cfg.data.seq_len


def test_type_coercion():
    cfg = load_config(None, ["train.compile=true", "train.betas=(0.9, 0.99)", "model.mlp_ratio=8"])
    assert cfg.train.compile is True
    assert cfg.train.betas == (0.9, 0.99)
    assert isinstance(cfg.model.mlp_ratio, float) and cfg.model.mlp_ratio == 8.0




def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(None, ["train.lr_typo=1e-3"])


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
