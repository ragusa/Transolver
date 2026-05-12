"""Utilities for 2D heat-conduction full-order model generation."""

from .fem_solver import (
    assemble_affine_components,
    load_mesh_and_tags,
    solve_for_parameters,
)
from .gmsh_mesh import generate_two_material_square_mesh

__all__ = [
    "assemble_affine_components",
    "generate_two_material_square_mesh",
    "load_mesh_and_tags",
    "solve_for_parameters",
]
