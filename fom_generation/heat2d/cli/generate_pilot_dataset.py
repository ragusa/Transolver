"""Generate a small Transolver-oriented pilot dataset for heat2d."""

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "heat2d_pilot_v1"
INPUT_FEATURE_NAMES = [
    "kappa_node",
    "q_node",
    "material_2_fraction",
    "outer_boundary_mask",
    "kappa_1",
    "kappa_2",
    "q_1",
    "q_2",
]


def generate_pilot_dataset(
    out_dir="fom_generation/data/heat2d_pilot",
    n_samples=6,
    seed=424242,
    mesh_size=0.11,
    safety_margin=0.06,
    plots_per_shape=1,
):
    """Generate a tiny per-sample NPZ dataset plus a manifest."""
    from ..fem_solver import assemble_affine_components, load_mesh_and_tags, solve_for_parameters
    from ..gmsh_mesh import generate_two_material_square_mesh, sample_random_inclusion_parameters

    if n_samples < 3:
        raise ValueError("n_samples must be at least 3 to cover disk, square, and triangle.")

    out_dir = Path(out_dir)
    samples_dir = out_dir / "samples"
    plots_dir = out_dir / "plots"
    samples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    shape_sequence = _shape_sequence(n_samples, rng)
    plot_counts = {"disk": 0, "square": 0, "triangle": 0}
    samples = []

    for sample_id, shape in enumerate(shape_sequence):
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
                "split": "train" if sample_id < max(1, int(0.8 * n_samples)) else "test",
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
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    _print_summary(manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a small heat2d pilot dataset.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_pilot")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--mesh-size", type=float, default=0.11)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    parser.add_argument("--plots-per-shape", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generate_pilot_dataset(
        out_dir=args.out,
        n_samples=args.n_samples,
        seed=args.seed,
        mesh_size=args.mesh_size,
        safety_margin=args.safety_margin,
        plots_per_shape=args.plots_per_shape,
    )


def _build_manifest(out_dir, samples, seed, mesh_size, safety_margin, plot_counts):
    train = [sample["sample_name"] for sample in samples if sample["split"] == "train"]
    test = [sample["sample_name"] for sample in samples if sample["split"] == "test"]
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Pilot heat-conduction FOM dataset for Transolver-oriented loading.",
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
        "splits": {"train": train, "test": test},
        "samples": samples,
        "plot_counts": plot_counts,
        "variable_size_meshes": True,
        "transolver_note": (
            "Existing fixed-grid benchmark loaders batch equal node counts. "
            "This pilot preserves variable meshes per sample; training can use "
            "batch_size=1, a padding/mask collate function, PyG-style batching, "
            "or later resampling to fixed query points."
        ),
    }


def _shape_sequence(n_samples, rng):
    shapes = ["disk", "square", "triangle"]
    extra = list(rng.choice(shapes, size=n_samples - len(shapes)))
    sequence = shapes + extra
    rng.shuffle(sequence)
    return sequence


def _sample_physical_parameters(rng):
    return {
        "kappa_1": float(rng.uniform(0.5, 5.0)),
        "kappa_2": float(rng.uniform(0.5, 20.0)),
        "q_1": float(rng.uniform(0.0, 10.0)),
        "q_2": float(rng.uniform(0.0, 10.0)),
    }


def _build_node_features(mesh_data, physical):
    triangles = mesh_data.triangles
    material_id = mesh_data.material_id
    n_nodes = mesh_data.coordinates.shape[0]
    element_kappa = np.where(material_id == 1, physical["kappa_1"], physical["kappa_2"])
    element_q = np.where(material_id == 1, physical["q_1"], physical["q_2"])
    element_mat2 = (material_id == 2).astype(float)

    counts = np.zeros(n_nodes, dtype=float)
    kappa_node = np.zeros(n_nodes, dtype=float)
    q_node = np.zeros(n_nodes, dtype=float)
    material_2_fraction = np.zeros(n_nodes, dtype=float)
    for local in range(3):
        nodes = triangles[:, local]
        np.add.at(counts, nodes, 1.0)
        np.add.at(kappa_node, nodes, element_kappa)
        np.add.at(q_node, nodes, element_q)
        np.add.at(material_2_fraction, nodes, element_mat2)

    counts = np.maximum(counts, 1.0)
    kappa_node /= counts
    q_node /= counts
    material_2_fraction /= counts

    boundary_mask = np.zeros(n_nodes, dtype=float)
    boundary_mask[mesh_data.outer_boundary_nodes] = 1.0
    nodal_material_id = np.where(material_2_fraction >= 0.5, 2, 1).astype(np.int64)

    repeated = {
        "kappa_1": np.full(n_nodes, physical["kappa_1"], dtype=float),
        "kappa_2": np.full(n_nodes, physical["kappa_2"], dtype=float),
        "q_1": np.full(n_nodes, physical["q_1"], dtype=float),
        "q_2": np.full(n_nodes, physical["q_2"], dtype=float),
    }
    node_features = np.column_stack(
        [
            kappa_node,
            q_node,
            material_2_fraction,
            boundary_mask,
            repeated["kappa_1"],
            repeated["kappa_2"],
            repeated["q_1"],
            repeated["q_2"],
        ]
    )
    return node_features.astype(np.float32), nodal_material_id


def _edge_index_from_triangles(triangles):
    edges = set()
    for a, b, c in triangles:
        for i, j in [(a, b), (b, c), (c, a)]:
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))
    edge_array = np.array(sorted(edges), dtype=np.int64)
    return edge_array.T if edge_array.size else np.zeros((2, 0), dtype=np.int64)


def _save_npz_sample(
    path,
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
):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape_type_id = {"disk": 0, "square": 1, "triangle": 2}[shape]
    np.savez_compressed(
        path,
        pos=mesh_data.coordinates.astype(np.float32),
        node_features=node_features,
        input_feature_names=np.array(INPUT_FEATURE_NAMES),
        target=temperature[:, None].astype(np.float32),
        coordinates=mesh_data.coordinates.astype(np.float32),
        triangles=mesh_data.triangles.astype(np.int64),
        element_material_id=mesh_data.material_id.astype(np.int64),
        nodal_material_id=nodal_material_id.astype(np.int64),
        edge_index=edge_index.astype(np.int64),
        kappa_1=np.array(physical["kappa_1"], dtype=np.float32),
        kappa_2=np.array(physical["kappa_2"], dtype=np.float32),
        q_1=np.array(physical["q_1"], dtype=np.float32),
        q_2=np.array(physical["q_2"], dtype=np.float32),
        shape_type=np.array(shape),
        shape_type_id=np.array(shape_type_id, dtype=np.int64),
        geometry_metadata_json=np.array(json.dumps(geometry_metadata)),
        mesh_filename=np.array(str(mesh_path)),
        sample_id=np.array(sample_id, dtype=np.int64),
        boundary_condition=np.array("T=0 on outer_boundary"),
        pde_sign_convention=np.array("-div(kappa grad T)=q"),
    )


def _save_plots(mesh_data, temperature, material_path, solution_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..visualize import plot_mesh_with_materials, plot_solution

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    material_plot = plot_mesh_with_materials(mesh_data, ax=ax)
    fig.colorbar(material_plot, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(material_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    solution_plot = plot_solution(mesh_data, temperature, ax=ax)
    fig.colorbar(solution_plot, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(solution_path, dpi=150)
    plt.close(fig)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _relative_to(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")


def _print_summary(manifest):
    print("Heat2D pilot dataset generated")
    print(f"root: {manifest['root']}")
    print(f"samples: {len(manifest['samples'])}")
    print(f"splits: train={len(manifest['splits']['train'])} test={len(manifest['splits']['test'])}")
    print("sample       shape       nodes    elems    split")
    for sample in manifest["samples"]:
        print(
            f"{sample['sample_name']:<12} "
            f"{sample['shape']:<11} "
            f"{sample['n_nodes']:<8d} "
            f"{sample['n_elements']:<8d} "
            f"{sample['split']}"
        )


if __name__ == "__main__":
    main()
