"""Smoke test for heat2d mesh generation, affine assembly, and solves."""

from pathlib import Path

import numpy as np

from .fem_solver import MATERIAL_1, MATERIAL_2, assemble_affine_components, load_mesh_and_tags, solve_for_parameters
from .gmsh_mesh import generate_two_material_square_mesh


def run_smoke_test(out_dir=None):
    if out_dir is None:
        base = Path("data") / "heat2d_smoke"
    else:
        base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    for i, shape in enumerate(["disk", "square", "triangle"]):
        mesh_path = base / shape / "mesh.msh"
        generate_two_material_square_mesh(
            mesh_path,
            shape=shape,
            outer_length=1.0,
            center=(0.5, 0.5),
            radius=0.18 if shape == "disk" else None,
            side_length=0.30 if shape != "disk" else None,
            rotation=0.2 * i,
            mesh_size_background=0.08,
            mesh_size_interface=0.04,
            seed=100 + i,
        )

        mesh_data = load_mesh_and_tags(mesh_path)
        affine = assemble_affine_components(mesh_data)
        T = solve_for_parameters(
            affine,
            kappa_1=1.0,
            kappa_2=10.0,
            q_1=1.0,
            q_2=5.0,
        )

        assert T.shape[0] == mesh_data.coordinates.shape[0]
        assert np.allclose(T[affine.dirichlet_dofs], 0.0, atol=1e-10)
        assert MATERIAL_1 in mesh_data.material_id
        assert MATERIAL_2 in mesh_data.material_id
        assert affine.last_relative_residual_norm < 1e-8

    print("heat2d smoke test passed")


if __name__ == "__main__":
    run_smoke_test()
