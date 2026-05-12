"""Visual audit for randomized two-material inclusion geometries."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .fem_solver import MATERIAL_1, MATERIAL_2, load_mesh_and_tags
from .gmsh_mesh import generate_two_material_square_mesh, sample_random_inclusion_parameters
from .visualize import plot_mesh_with_materials


def run_validation(
    out_dir="fom_generation/data/heat2d_geometry_randomization",
    examples_per_shape=3,
    seed=12345,
    mesh_size=0.09,
    safety_margin=0.06,
):
    """Generate randomized geometries and save material-ID audit plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results = []

    for shape in ["disk", "square", "triangle"]:
        for case_id in range(examples_per_shape):
            params = sample_random_inclusion_parameters(
                shape,
                outer_length=1.0,
                safety_margin=safety_margin,
                rng=rng,
            )
            _check_rotation(shape, params["rotation"])

            case_dir = out_dir / shape / f"case_{case_id:03d}"
            mesh_path = case_dir / "mesh.msh"
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
            counts = _material_counts(mesh_data.material_id)
            _check_material_counts(counts, shape, case_id)

            plot_path = case_dir / "material_ids.png"
            _save_material_plot(mesh_data, plot_path)

            result = {
                "shape": shape,
                "case_id": int(case_id),
                "center": [float(params["center"][0]), float(params["center"][1])],
                "radius": None if params["radius"] is None else float(params["radius"]),
                "side_length": None if params["side_length"] is None else float(params["side_length"]),
                "rotation": float(params["rotation"]),
                "safety_margin": float(safety_margin),
                "n_nodes": int(mesh_data.coordinates.shape[0]),
                "n_elements": int(mesh_data.triangles.shape[0]),
                "material_1_elements": int(counts[MATERIAL_1]),
                "material_2_elements": int(counts[MATERIAL_2]),
                "mesh_path": str(mesh_path),
                "plot_path": str(plot_path),
            }
            results.append(result)

    summary = {
        "seed": int(seed),
        "mesh_size": float(mesh_size),
        "safety_margin": float(safety_margin),
        "examples_per_shape": int(examples_per_shape),
        "all_cases_passed": True,
        "results": results,
    }
    with (out_dir / "geometry_randomization_validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    _print_summary(summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Visual audit for randomized heat2d geometries.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_geometry_randomization")
    parser.add_argument("--examples-per-shape", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mesh-size", type=float, default=0.09)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_validation(
        out_dir=args.out,
        examples_per_shape=args.examples_per_shape,
        seed=args.seed,
        mesh_size=args.mesh_size,
        safety_margin=args.safety_margin,
    )


def _material_counts(material_id):
    labels, counts = np.unique(material_id, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _check_material_counts(counts, shape, case_id):
    for material in [MATERIAL_1, MATERIAL_2]:
        if material not in counts:
            raise AssertionError(f"{shape} case {case_id}: missing material tag {material}")
        if counts[material] <= 0:
            raise AssertionError(f"{shape} case {case_id}: material {material} has no elements")


def _check_rotation(shape, rotation):
    if shape == "disk":
        return
    distance_to_axis = min(abs(rotation - k * math.pi / 2.0) for k in range(3))
    if distance_to_axis <= 0.08:
        raise AssertionError(f"{shape}: sampled rotation is too close to axis aligned")


def _save_material_plot(mesh_data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    material_plot = plot_mesh_with_materials(mesh_data, ax=ax)
    fig.colorbar(material_plot, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_summary(summary):
    print("Geometry randomization/material-ID visual audit")
    print(
        f"seed={summary['seed']} mesh_size={summary['mesh_size']} "
        f"safety_margin={summary['safety_margin']}"
    )
    print("shape       case    elems    mat1    mat2    center              size        rotation")
    for result in summary["results"]:
        size = result["radius"] if result["radius"] is not None else result["side_length"]
        center = f"({result['center'][0]:.3f}, {result['center'][1]:.3f})"
        print(
            f"{result['shape']:<11} "
            f"{result['case_id']:<7d} "
            f"{result['n_elements']:<8d} "
            f"{result['material_1_elements']:<7d} "
            f"{result['material_2_elements']:<7d} "
            f"{center:<19} "
            f"{size:<11.4f} "
            f"{result['rotation']:.4f}"
        )
    print(f"All cases passed: {summary['all_cases_passed']}")


if __name__ == "__main__":
    main()
