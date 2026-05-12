"""Validate two-material mesh labels and solves for supported inclusions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .fem_solver import MATERIAL_1, MATERIAL_2, assemble_affine_components, load_mesh_and_tags, solve_for_parameters
from .gmsh_mesh import generate_two_material_square_mesh
from .visualize import plot_mesh_with_materials, plot_solution


def run_validation(
    out_dir="fom_generation/data/heat2d_material_validation",
    shapes=("disk", "square", "triangle"),
    mesh_size=0.07,
    kappa_1=1.0,
    kappa_2=10.0,
    q_value=1.0,
):
    """Generate, label-check, solve, and plot one case per inclusion shape."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, shape in enumerate(shapes):
        shape = shape.lower()
        case_dir = out_dir / shape
        mesh_path = case_dir / "mesh.msh"
        generate_two_material_square_mesh(
            mesh_path,
            shape=shape,
            outer_length=1.0,
            center=(0.5, 0.5),
            radius=0.18 if shape == "disk" else None,
            side_length=0.30 if shape != "disk" else None,
            rotation=0.25 if shape != "disk" else 0.0,
            mesh_size_background=mesh_size,
            mesh_size_interface=0.5 * mesh_size,
            seed=3000 + index,
        )

        mesh_data = load_mesh_and_tags(mesh_path)
        material_counts = _material_counts(mesh_data.material_id)
        _check_material_counts(material_counts, shape)

        affine = assemble_affine_components(mesh_data)
        temperature = solve_for_parameters(
            affine,
            kappa_1=kappa_1,
            kappa_2=kappa_2,
            q_1=q_value,
            q_2=q_value,
        )
        _check_solution(mesh_data, affine, temperature, shape)

        plot_path = case_dir / "material_and_temperature.png"
        _save_diagnostic_plot(mesh_data, temperature, plot_path)

        result = {
            "shape": shape,
            "mesh_path": str(mesh_path),
            "plot_path": str(plot_path),
            "n_nodes": int(mesh_data.coordinates.shape[0]),
            "n_elements": int(mesh_data.triangles.shape[0]),
            "material_1_elements": int(material_counts[MATERIAL_1]),
            "material_2_elements": int(material_counts[MATERIAL_2]),
            "n_dirichlet_dofs": int(affine.dirichlet_dofs.size),
            "kappa_1": float(kappa_1),
            "kappa_2": float(kappa_2),
            "q_1": float(q_value),
            "q_2": float(q_value),
            "min_temperature": float(temperature.min()),
            "max_temperature": float(temperature.max()),
            "relative_residual": affine.last_relative_residual_norm,
        }
        results.append(result)

    summary = {
        "pde_form": "-div(kappa grad T) = q",
        "material_1": "background",
        "material_2": "inclusion",
        "outer_boundary": "homogeneous Dirichlet",
        "all_shapes_passed": True,
        "results": results,
    }
    with (out_dir / "material_validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    _print_summary(summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Validate two-material heat2d meshes and solves.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_material_validation")
    parser.add_argument("--mesh-size", type=float, default=0.07)
    parser.add_argument("--kappa-1", type=float, default=1.0)
    parser.add_argument("--kappa-2", type=float, default=10.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--shapes", nargs="+", default=["disk", "square", "triangle"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_validation(
        out_dir=args.out,
        shapes=args.shapes,
        mesh_size=args.mesh_size,
        kappa_1=args.kappa_1,
        kappa_2=args.kappa_2,
        q_value=args.q,
    )


def _material_counts(material_id):
    labels, counts = np.unique(material_id, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _check_material_counts(material_counts, shape):
    for material in [MATERIAL_1, MATERIAL_2]:
        if material not in material_counts:
            raise AssertionError(f"{shape}: missing material tag {material}")
        if material_counts[material] <= 0:
            raise AssertionError(f"{shape}: material tag {material} has no elements")


def _check_solution(mesh_data, affine, temperature, shape):
    if temperature.shape[0] != mesh_data.coordinates.shape[0]:
        raise AssertionError(f"{shape}: solution length does not match node count")
    if not np.all(np.isfinite(temperature)):
        raise AssertionError(f"{shape}: solution contains non-finite values")
    if not np.allclose(temperature[affine.dirichlet_dofs], 0.0, atol=1e-10):
        raise AssertionError(f"{shape}: nonzero values on outer Dirichlet boundary")
    if affine.last_relative_residual_norm > 1e-8:
        raise AssertionError(f"{shape}: large relative residual {affine.last_relative_residual_norm:.3e}")
    if temperature.max() <= 0.0:
        raise AssertionError(f"{shape}: positive source did not produce a positive temperature maximum")


def _save_diagnostic_plot(mesh_data, temperature, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    material_plot = plot_mesh_with_materials(mesh_data, ax=axes[0])
    solution_plot = plot_solution(mesh_data, temperature, ax=axes[1])
    fig.colorbar(material_plot, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(solution_plot, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_summary(summary):
    print("Two-material geometry/material validation")
    print(f"PDE form: {summary['pde_form']}")
    print("shape       elems    mat1    mat2    bnd_dofs    Tmax         rel_residual")
    for result in summary["results"]:
        print(
            f"{result['shape']:<11} "
            f"{result['n_elements']:<8d} "
            f"{result['material_1_elements']:<7d} "
            f"{result['material_2_elements']:<7d} "
            f"{result['n_dirichlet_dofs']:<11d} "
            f"{result['max_temperature']:<12.6e} "
            f"{result['relative_residual']:.3e}"
        )
    print(f"All shapes passed: {summary['all_shapes_passed']}")


if __name__ == "__main__":
    main()
