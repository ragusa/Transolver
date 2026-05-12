"""Configuration helpers for heat2d FOM dataset generation."""

import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "n_geometries": 3,
    "n_params_per_geometry": 2,
    "seed": 12345,
    "outer_length": 1.0,
    "mesh_size_background": 0.05,
    "mesh_size_interface": 0.025,
    "allowed_shapes": ["disk", "square", "triangle"],
    "parameter_ranges": {
        "kappa_1": [0.5, 5.0],
        "kappa_2": [0.5, 20.0],
        "q_1": [0.0, 10.0],
        "q_2": [0.0, 10.0],
    },
}


def default_config():
    """Return a mutable copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path=None):
    """Load a JSON config file, falling back to defaults when path is None."""
    cfg = default_config()
    if path is None:
        return cfg

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML configs require PyYAML. Use config_default.json "
                    "or install pyyaml."
                ) from exc
            user_cfg = yaml.safe_load(f) or {}
        else:
            user_cfg = json.load(f)

    _deep_update(cfg, user_cfg)
    return cfg


def save_config(config, path):
    """Save a config dictionary as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def apply_overrides(config, args):
    """Apply argparse overrides in-place and return config."""
    if getattr(args, "n_geometries", None) is not None:
        config["n_geometries"] = args.n_geometries
    if getattr(args, "n_params_per_geometry", None) is not None:
        config["n_params_per_geometry"] = args.n_params_per_geometry
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if getattr(args, "mesh_size", None) is not None:
        config["mesh_size_background"] = args.mesh_size
        config["mesh_size_interface"] = 0.5 * args.mesh_size
    return config


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
