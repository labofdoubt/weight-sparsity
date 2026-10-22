import os

import pytest
import yaml

from toy_mnist.config import Config, load_config

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
SHIPPED = ["baseline", "rescale", "topk_hard", "topk_dust", "topk_dust_backprop"]


def test_defaults_validate():
    cfg = Config().validate()
    assert cfg.topk.dust_schedule == [1, 1]
    assert set(Config.DUST_DEFAULTS) == {k for k in vars(cfg.topk) if k.startswith("dust_")}


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(CONFIG_DIR, f"{name}.yaml"))
    assert cfg.train.run_name == name
    if name.startswith("topk_dust"):
        assert cfg.topk.surrogate_mode == "dust" and cfg.rescale.enabled
        assert cfg.topk.dust_target == ("backprop" if name.endswith("backprop") else "encoder")
        assert cfg.topk.j is None and cfg.topk.resolved_j() == 784 - cfg.topk.k
    if name == "topk_hard":
        assert cfg.topk.j is None and cfg.topk.resolved_j() == 0


def test_pool_width_default_and_override():
    base = {"rescale": {"enabled": True}, "topk": {"enabled": True, "k": 16}}
    assert load_config(values=base).topk.resolved_j() == 0
    dust = {"rescale": {"enabled": True},
            "topk": {"enabled": True, "k": 16, "surrogate_mode": "dust"}}
    assert load_config(values=dust).topk.resolved_j() == 768
    cfg = load_config(values=dust, overrides=["topk.j=5"])
    assert cfg.topk.j == 5 and cfg.topk.resolved_j() == 5
    cfg = load_config(values=dust, overrides=["topk.j=null"])
    assert cfg.topk.j is None
    with pytest.raises(ValueError, match="topk.j"):
        load_config(values=dust, overrides=["topk.j=-1"])


def test_base_includes_and_overrides(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"train": {"lr": 0.5, "seed": 3}}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(
        {"_base_": ["a.yaml"], "train": {"seed": 4}, "rescale": {"enabled": True}}))
    cfg = load_config(str(tmp_path / "b.yaml"),
                      overrides=["train.lr=1e-2", "--topk.enabled=true", "topk.k=5",
                                 "train.betas=[0.8, 0.9]", "train.tensorboard=false"])
    assert cfg.train.lr == 1e-2 and cfg.train.seed == 4 and cfg.rescale.enabled
    assert cfg.topk.enabled and cfg.topk.k == 5
    assert cfg.train.betas == [0.8, 0.9] and cfg.train.tensorboard is False


@pytest.mark.parametrize("values, message", [
    ({"topk": {"dust_sigma": 0.3}}, "dust_\\* fields apply"),
    ({"topk": {"enabled": True, "surrogate_mode": "dust"}}, "rescale.enabled"),
    ({"rescale": {"enabled": True}, "topk": {"surrogate_mode": "dust"}}, "topk.enabled"),
    ({"rescale": {"enabled": True}, "topk": {"enabled": True, "surrogate_mode": "dust",
                                             "dust_gamma": 0.5}}, "dust_gamma must be 0"),
    ({"rescale": {"enabled": True}, "topk": {"enabled": True, "surrogate_mode": "dust",
                                             "dust_population": 1}}, "dust_population"),
    ({"rescale": {"enabled": True}, "topk": {"enabled": True, "surrogate_mode": "dust",
                                             "dust_schedule": [1, 0]}}, "n_dust >= 1"),
    ({"rescale": {"enabled": True}, "topk": {"enabled": True, "surrogate_mode": "dust",
                                             "dust_target": "decoder"}}, "dust_target"),
    ({"topk": {"enabled": True, "k": 0}}, "topk.k"),
    ({"nope": {"x": 1}}, "unknown config key"),
    ({"train": {"lr": "fast"}}, "could not convert"),
])
def test_invalid_configurations_are_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        load_config(values=values)
