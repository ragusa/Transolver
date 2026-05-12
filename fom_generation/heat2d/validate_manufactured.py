"""Manufactured-solution validation for the heat2d FEM solver."""

import argparse
import json
from pathlib import Path

import numpy as np

from .fem_solver import assemble_affine_components, l2_error, load_mesh_and_tags, solve_for_callable_source
from .gmsh_mesh import generate_two_material_square_mesh


def exact_solution(x, y):
    """T(x, y) = sin(pi x) sin(pi y), zero on the unit-square boundary."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def manufactured_source(x, y, kappa=1.0):
    """Source for -div(kappa grad T) = q with constant kappa."""
    return 2.0 * kappa * np.pi**2 * exact_solution(x, y)


def run_validation(mesh_sizes=(0.12, 0.06), out_dir="fom_generation/data/heat2d_validation", kappa=1.0):
    """Run the manufactured-solution check on a sequence of mesh sizes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, mesh_size in enumerate(mesh_sizes):
        case_dir = out_dir / f"h_{mesh_size:g}".replace(".", "p")
        mesh_path = case_dir / "mesh.msh"
        generate_two_material_square_mesh(
            mesh_path,
            shape="disk",
            outer_length=1.0,
            center=(0.5, 0.5),
            radius=0.18,
            mesh_size_background=float(mesh_size),
            mesh_size_interface=0.5 * float(mesh_size),
            seed=2000 + index,
        )

        mesh_data = load_mesh_and_tags(mesh_path)
        affine = assemble_affine_components(mesh_data)

        # Same conductivity in both physical regions makes the PDE homogeneous;
        # the internal interface remains in the mesh but has no physical jump.
        source = lambda x, y: manufactured_source(x, y, kappa=kappa)
        numerical = solve_for_callable_source(affine, kappa, kappa, source)
        exact = exact_solution(mesh_data.coordinates[:, 0], mesh_data.coordinates[:, 1])
        abs_l2, rel_l2 = l2_error(affine.basis, numerical, exact)

        result = {
            "mesh_size": float(mesh_size),
            "n_nodes": int(mesh_data.coordinates.shape[0]),
            "n_elements": int(mesh_data.triangles.shape[0]),
            "absolute_l2_error": abs_l2,
            "relative_l2_error": rel_l2,
            "relative_residual": affine.last_relative_residual_norm,
            "mesh_path": str(mesh_path),
        }
        results.append(result)

    decreases = all(
        later["relative_l2_error"] < earlier["relative_l2_error"]
        for earlier, later in zip(results, results[1:])
    )
    summary = {
        "pde_form": "-div(kappa grad T) = q",
        "exact_solution": "sin(pi x) sin(pi y)",
        "source_term": "q = 2 kappa pi^2 sin(pi x) sin(pi y)",
        "kappa": float(kappa),
        "error_decreases": bool(decreases),
        "results": results,
    }
    with (out_dir / "manufactured_validation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    _print_summary(summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Validate heat2d with a manufactured solution.")
    parser.add_argument("--out", default="fom_generation/data/heat2d_validation")
    parser.add_argument("--mesh-sizes", nargs="+", type=float, default=[0.12, 0.06])
    parser.add_argument("--kappa", type=float, default=1.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_validation(mesh_sizes=args.mesh_sizes, out_dir=args.out, kappa=args.kappa)


def _print_summary(summary):
    print("Manufactured solution validation")
    print(f"PDE form: {summary['pde_form']}")
    print(f"Source: {summary['source_term']}")
    print("mesh_size    nodes    elems    rel_L2_error    rel_residual")
    for result in summary["results"]:
        print(
            f"{result['mesh_size']:<11g} "
            f"{result['n_nodes']:<8d} "
            f"{result['n_elements']:<8d} "
            f"{result['relative_l2_error']:<15.6e} "
            f"{result['relative_residual']:.3e}"
        )
    print(f"Error decreases: {summary['error_decreases']}")


if __name__ == "__main__":
    main()
