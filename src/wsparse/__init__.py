"""wsparse -- TinyStories language models with activation bottlenecks."""

from .config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from .model import TransformerLM, build_model

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "load_config",
    "TransformerLM",
    "build_model",
]
