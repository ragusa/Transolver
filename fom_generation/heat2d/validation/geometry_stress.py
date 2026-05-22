"""Lightweight randomized geometry stress test for heat2d meshes."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

MATERIAL_1 = 1
MATERIAL_2 = 2


def run_validation(
    out_dir="fom_generation/data/heat2d_geometry_stress",
    n_geometries=36,
    seed=20260512,
    mesh_size=0.11,
    safety_margin=0.06,
    plots_per_shape=3,
):
    """Generate many random geometries and verify material labels after import."""
    from ..fem_solver import load_mesh_and_tags
    from ..gmsh_mesh import generate_two_material_square_mesh, sample_random_inclusion_parameters

    if n_geometries < 3:
        raise ValueError("n_geometries must be at least 3 to cover all supported shapes.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shapes = _shape_sequence(n_geometries, rng)
    plot_counts = {"disk": 0, "square": 0, "triangle": 0}
    results = []
    total_rejections = 0

    for case_id, shape in enumerate(shapes):
        params = sample_random_inclusion_parameters(
            shape,
            outer_length=1.0,
            safety_margin=safety_margin,
            rng=rng,
            include_attempts=True,
        )
        _check_rotation(shape, params["rotation"])
        total_rejections += params["n_rejections"]

        case_dir = out_dir / f"case_{case_id:04d}_{shape}"
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

        plot_path = None
        if plot_counts[shape] < plots_per_shape:
            plot_path = case_dir / "material_ids.png"
            _save_material_plot(mesh_data, plot_path)
            plot_counts[shape] += 1

        result = {
            "case_id": int(case_id),
            "shape": shape,
            "center": [float(params["center"][0]), float(params["center"][1])],
            "radius": None if params["radius"] is None else float(params["radius"]),
            "side_length": None if params["side_length"] is None else float(params["side_length"]),
            "rotation": float(params["rotation"]),
            "n_attempts": int(params["n_attempts"]),
            "n_rejections": int(params["n_rejections"]),
            "n_nodes": int(mesh_data.coordinates.shape[0]),
            "n_elements": int(mesh_data.triangles.shape[0]),
            "material_1_elements": int(counts[MATERIAL_1]),
            "material_2_elements": int(counts[MATERIAL_2]),
            "mesh_path": str(mesh_path),
            "plot_path": None if plot_path is None else str(plot_path),
        }
        results.append(result)

    shape_counts = {shape: shapes.count(shape) for shape in ["disk", "square", "triangle"]}
    saved_plot_counts = dict(plot_counts)
    summary = {
        "seed": int(seed),
        "n_geometries_requested": int(n_geometries),
        "n_geometries_passed": int(len(results)),
        "total_rejections_before_gmsh": int(total_rejections),
        "mesh_size": float(mesh_size),
        "safety_margin": float(safety_margin),
        "plots_per_shape": int(plots_per_shape),
        "shape_counts": shape_counts,
        "saved_plot_counts": saved_plot_counts,
        "all_cases_passed": True,
        "results": results,
    }
    with (out_dir / "geometry_stress_validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    _print_summary(summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Stress test randomized heat2d geometries.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_geometry_stress")
    parser.add_argument("--n-geometries", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--mesh-size", type=float, default=0.11)
    parser.add_argument("--safety-margin", type=float, default=0.06)
    parser.add_argument("--plots-per-shape", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_validation(
        out_dir=args.out,
        n_geometries=args.n_geometries,
        seed=args.seed,
        mesh_size=args.mesh_size,
        safety_margin=args.safety_margin,
        plots_per_shape=args.plots_per_shape,
    )


def _shape_sequence(n_geometries, rng):
    shapes = ["disk", "square", "triangle"]
    extra = list(rng.choice(shapes, size=n_geometries - len(shapes)))
    sequence = shapes + extra
    rng.shuffle(sequence)
    return sequence


def _material_counts(material_id):
    labels, counts = np.unique(material_id, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _check_material_counts(counts, shape, case_id):
    for material in [MATERIAL_1, MATERIAL_2]:
        if material not in counts:
            raise AssertionError(f"case {case_id} {shape}: missing material tag {material}")
        if counts[material] <= 0:
            raise AssertionError(f"case {case_id} {shape}: material {material} has no elements")


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

    from ..visualize import plot_mesh_with_materials

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    material_plot = plot_mesh_with_materials(mesh_data, ax=ax)
    fig.colorbar(material_plot, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_summary(summary):
    print("Randomized geometry stress validation")
    print(
        f"seed={summary['seed']} n={summary['n_geometries_requested']} "
        f"mesh_size={summary['mesh_size']} safety_margin={summary['safety_margin']}"
    )
    print(f"shape counts: {summary['shape_counts']}")
    print(f"saved plot counts: {summary['saved_plot_counts']}")
    print(f"pre-Gmsh rejected/resampled candidates: {summary['total_rejections_before_gmsh']}")
    print("case    shape       elems    mat1    mat2    attempts    plot")
    for result in summary["results"]:
        plot_flag = "yes" if result["plot_path"] else "no"
        print(
            f"{result['case_id']:<7d} "
            f"{result['shape']:<11} "
            f"{result['n_elements']:<8d} "
            f"{result['material_1_elements']:<7d} "
            f"{result['material_2_elements']:<7d} "
            f"{result['n_attempts']:<11d} "
            f"{plot_flag}"
        )
    print(f"Geometries passed: {summary['n_geometries_passed']}")
    print(f"All cases passed: {summary['all_cases_passed']}")


if __name__ == "__main__":
    main()
