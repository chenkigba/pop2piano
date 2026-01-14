"""Configuration management for Pop2Piano."""

from pathlib import Path
from omegaconf import OmegaConf

_CONFIG_DIR = Path(__file__).parent


def get_default_config():
    """Load the default configuration."""
    config_path = _CONFIG_DIR / "default.yaml"
    return OmegaConf.load(config_path)
