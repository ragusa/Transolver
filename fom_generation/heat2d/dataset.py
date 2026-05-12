"""Dataset generation for heat2d full-order model samples."""

import json
from pathlib import Path

import numpy as np

from .config import save_config
from .fem_solver import assemble_affine_components, load_mesh_and_tags, save_affine_info, solve_for_parameters
from .gmsh_mesh import generate_two_material_square_mesh


def generate_dataset(config, out_dir, plot_first=False):
    """Generate meshes and parameter-sweep solutions into compressed NPZ files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config_used.json")

    rng = np.random.default_rng(config.get("seed", 12345))
    allowed_shapes = config.get("allowed_shapes", ["disk", "square", "triangle"])
    n_geometries = int(config["n_geometries"])
    n_params = int(config["n_params_per_geometry"])
    sample_paths = []

    iterator = range(n_geometries)
    try:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="geometries")
    except ImportError:
        pass

    for geometry_id in iterator:
        geom_dir = out_dir / "geometries" / f"geom_{geometry_id:05d}"
        geom_dir.mkdir(parents=True, exist_ok=True)

        shape = str(rng.choice(allowed_shapes))
        mesh_path = geom_dir / "mesh.msh"
        generate_two_material_square_mesh(
            mesh_path,
            shape=shape,
            outer_length=float(config.get("outer_length", 1.0)),
            mesh_size_background=float(config.get("mesh_size_background", 0.05)),
            mesh_size_interface=float(config.get("mesh_size_interface", 0.025)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        mesh_data = load_mesh_and_tags(mesh_path)
        affine = assemble_affine_components(mesh_data)
        save_affine_info(affine, geom_dir / "affine_info.json")
        geometry_metadata = _load_json(mesh_path.with_name("mesh_metadata.json"))

        for param_id in range(n_params):
            params = _sample_parameters(config["parameter_ranges"], rng)
            T = solve_for_parameters(affine, **params)
            sample_path = geom_dir / f"sample_{param_id:05d}.npz"
            _save_sample(sample_path, geometry_id, param_id, mesh_data, T, params, geometry_metadata, mesh_path)
            sample_paths.append(sample_path)

            if plot_first and geometry_id == 0 and param_id == 0:
                _plot_first_sample(geom_dir, mesh_data, T)

    return sample_paths


def _sample_parameters(parameter_ranges, rng):
    params = {}
    for name in ["kappa_1", "kappa_2", "q_1", "q_2"]:
        lo, hi = parameter_ranges[name]
        params[name] = float(rng.uniform(lo, hi))
    return params


def _save_sample(sample_path, geometry_id, param_id, mesh_data, T, params, geometry_metadata, mesh_path):
    param_names = np.array(["kappa_1", "kappa_2", "q_1", "q_2"])
    param_values = np.array([params[name] for name in param_names], dtype=float)
    np.savez_compressed(
        sample_path,
        coordinates=mesh_data.coordinates,
        triangles=mesh_data.triangles,
        material_id=mesh_data.material_id,
        T=T,
        param_names=param_names,
        param_values=param_values,
        geometry_metadata_json=json.dumps(geometry_metadata),
        mesh_filename=str(mesh_path),
        sample_id=np.array(param_id, dtype=np.int64),
        geometry_id=np.array(geometry_id, dtype=np.int64),
    )


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_first_sample(geom_dir, mesh_data, T):
    import matplotlib.pyplot as plt

    from .visualize import plot_mesh_with_materials, plot_solution

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_mesh_with_materials(mesh_data, ax=ax)
    fig.tight_layout()
    fig.savefig(geom_dir / "mesh_materials.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_solution(mesh_data, T, ax=ax)
    fig.tight_layout()
    fig.savefig(geom_dir / "solution.png", dpi=150)
    plt.close(fig)
