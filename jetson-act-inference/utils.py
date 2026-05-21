"""Utility helpers: config loading, safetensors stats loading."""

from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import load_file


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_stats(safetensors_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load normalization stats from a safetensors file.

    The safetensors file stores flat keys like "observation.state.mean",
    "observation.state.std", "action.mean", "action.std", etc.

    Returns nested dict: {feature_key: {stat_name: ndarray}}.
    E.g. {"observation.state": {"mean": array, "std": array}, ...}
    """
    flat = load_file(safetensors_path)
    stats: dict[str, dict[str, np.ndarray]] = {}
    for flat_key, tensor in flat.items():
        # Split on last dot: "observation.state.mean" -> ("observation.state", "mean")
        key, stat_name = flat_key.rsplit(".", 1)
        if key not in stats:
            stats[key] = {}
        stats[key][stat_name] = tensor.astype(np.float32)
    return stats


def get_calibration_path(cfg_path: str) -> Path:
    """Return expanded calibration path from config."""
    return Path(cfg_path).expanduser()
