"""Generate a balanced Heat2D dataset using the established NPZ schema."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .check_dataset import check_dataset
from .generate_pilot_dataset import (
    INPUT_FEATURE_NAMES,
    SCHEMA_VERSION,
    _build_node_features,
    _edge_index_from_triangles,
    _load_json,
    _relative_to,
    _sample_physical_parameters,
    _save_npz_sample,
    _save_plots,
)


SHAPES = ("disk", "square", "triangle")


def generate_balanced_dataset(
    out_dir="fom_generation/data/heat2d_moderate_balanced",
    samples_per_shape=30,
    seed=20260512,
    mesh_size=0.14,
    safety_margin=0.06,
    plots_per_shape=1,
):
    from ..fem_solver import assemble_affine_components, load_mesh_and_tags, solve_for_parameters
    from ..gmsh_mesh import generate_two_material_square_mesh, sample_random_inclusion_parameters

    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists. Choose a new --out directory.")

    samples_dir = out_dir / "samples"
    plots_dir = out_dir / "plots"
    samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    split_by_shape = _stratified_split_by_shape(samples_per_shape, rng)
    shape_sequence = [(shape, split) for shape in SHAPES for split in split_by_shape[shape]]
    rng.shuffle(shape_sequence)

    samples = []
    plot_counts = {shape: 0 for shape in SHAPES}
    for sample_id, (shape, split) in enumerate(shape_sequence):
        params = sample_random_inclusion_parameters(
            shape,
            outer_length=1.0,
            safety_margin=safety_margin,
            rng=rng,
            include_attempts=True,
        )
        physical = _sample_physical_parameters(rng)

        sample_name = f"sample_{sample_id:05d}_{shape}"
        sample_dir = samples_dir / sample_name
        mesh_path = sample_dir / "mesh.msh"
        generate_two_material_square_mesh(
            mesh_path,
            shape=params["shape"],
            outer_length=1.0,
            center=params["center"],
            radius=params["radius"],
            side_length=params["side_length"],
            rotation=params["rotation"],
            mesh_size_background=mesh_size,
            mesh_size_interface=0.5 * mesh_size,
            safety_margin=safety_margin,
        )

        mesh_data = load_mesh_and_tags(mesh_path)
        affine = assemble_affine_components(mesh_data)
        temperature = solve_for_parameters(affine, **physical)
        geometry_metadata = _load_json(mesh_path.with_name("mesh_metadata.json"))

        node_features, nodal_material_id = _build_node_features(mesh_data, physical)
        edge_index = _edge_index_from_triangles(mesh_data.triangles)
        npz_path = sample_dir / "sample.npz"
        _save_npz_sample(
            npz_path,
            sample_id,
            shape,
            mesh_data,
            temperature,
            physical,
            geometry_metadata,
            node_features,
            nodal_material_id,
            edge_index,
            mesh_path,
        )

        material_plot = None
        solution_plot = None
        if plot_counts[shape] < plots_per_shape:
            plot_dir = plots_dir / sample_name
            plot_dir.mkdir(parents=True, exist_ok=True)
            material_plot = plot_dir / "material_ids.png"
            solution_plot = plot_dir / "temperature.png"
            _save_plots(mesh_data, temperature, material_plot, solution_plot)
            plot_counts[shape] += 1

        samples.append(
            {
                "sample_id": int(sample_id),
                "sample_name": sample_name,
                "file": _relative_to(npz_path, out_dir),
                "mesh_file": _relative_to(mesh_path, out_dir),
                "shape": shape,
                "split": split,
                "n_nodes": int(mesh_data.coordinates.shape[0]),
                "n_elements": int(mesh_data.triangles.shape[0]),
                "material_1_elements": int(np.sum(mesh_data.material_id == 1)),
                "material_2_elements": int(np.sum(mesh_data.material_id == 2)),
                "kappa_1": float(physical["kappa_1"]),
                "kappa_2": float(physical["kappa_2"]),
                "q_1": float(physical["q_1"]),
                "q_2": float(physical["q_2"]),
                "inclusion_center": [float(params["center"][0]), float(params["center"][1])],
                "inclusion_radius": None if params["radius"] is None else float(params["radius"]),
                "inclusion_side_length": None if params["side_length"] is None else float(params["side_length"]),
                "inclusion_rotation": float(params["rotation"]),
                "material_plot": None if material_plot is None else _relative_to(material_plot, out_dir),
                "solution_plot": None if solution_plot is None else _relative_to(solution_plot, out_dir),
            }
        )

    manifest = _build_manifest(out_dir, samples, seed, mesh_size, safety_margin, plot_counts)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    summary = check_dataset(out_dir)
    _print_summary(summary)
    return manifest, summary


def _stratified_split_by_shape(samples_per_shape, rng):
    train_count = int(round(0.7 * samples_per_shape))
    val_count = int(round(0.1 * samples_per_shape))
    test_count = samples_per_shape - train_count - val_count
    splits = {}
    for shape in SHAPES:
        shape_splits = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        rng.shuffle(shape_splits)
        splits[shape] = shape_splits
    return splits


def _build_manifest(out_dir, samples, seed, mesh_size, safety_margin, plot_counts):
    splits = {
        split: [sample["sample_name"] for sample in samples if sample["split"] == split]
        for split in ["train", "val", "test"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Balanced moderate heat-conduction FOM dataset for Transolver training.",
        "storage_format": "one compressed NPZ file per sample plus this manifest",
        "root": str(out_dir),
        "seed": int(seed),
        "mesh_size": float(mesh_size),
        "safety_margin": float(safety_margin),
        "pde": {
            "strong_form": "-div(kappa grad T) = q",
            "boundary_condition": "T = 0 on outer_boundary",
            "materials": {"1": "background", "2": "inclusion"},
            "outer_boundary_tag": 101,
        },
        "arrays": {
            "pos": "(n_nodes, 2) node coordinates; Transolver position input",
            "node_features": "(n_nodes, 8) nodewise material/source/parameter features",
            "input_feature_names": INPUT_FEATURE_NAMES,
            "target": "(n_nodes, 1) nodal temperature",
            "coordinates": "(n_nodes, 2) duplicate of pos for explicit FEM naming",
            "triangles": "(n_elements, 3) triangular connectivity",
            "element_material_id": "(n_elements,) physical material tag per element",
            "nodal_material_id": "(n_nodes,) rounded node material tag for quick checks",
            "edge_index": "(2, n_edges) directed triangle-edge graph, optional for graph loaders",
        },
        "splits": splits,
        "samples": samples,
        "plot_counts": plot_counts,
        "shape_counts": dict(Counter(sample["shape"] for sample in samples)),
        "variable_size_meshes": True,
        "transolver_note": (
            "This dataset preserves variable triangular meshes per sample. "
            "Current Transolver integration intentionally uses batch_size=1."
        ),
    }


def _print_summary(summary):
    print("Heat2D balanced dataset generated and checked")
    print(f"root: {summary['output_directory']}")
    print(f"samples: {summary['num_samples']}")
    print(f"shape_counts: {summary['shape_counts']}")
    print(f"split_counts: {summary['split_counts']}")
    print(f"node_count: {summary['node_count']}")
    print(f"element_count: {summary['element_count']}")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a balanced Heat2D NPZ dataset.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_moderate_balanced")
    parser.add_argument("--samples-per-shape", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--mesh-size", type=float, default=0.14)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    parser.add_argument("--plots-per-shape", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generate_balanced_dataset(
        out_dir=args.out,
        samples_per_shape=args.samples_per_shape,
        seed=args.seed,
        mesh_size=args.mesh_size,
        safety_margin=args.safety_margin,
        plots_per_shape=args.plots_per_shape,
    )


if __name__ == "__main__":
    main()
