"""Utilities for 2D heat-conduction full-order model generation."""

from .fem_solver import (
    assemble_affine_components,
    assemble_callable_source_vector,
    l2_error,
    load_mesh_and_tags,
    solve_for_callable_source,
    solve_for_parameters,
)
from .gmsh_mesh import generate_two_material_square_mesh, sample_random_inclusion_parameters

__all__ = [
    "assemble_affine_components",
    "assemble_callable_source_vector",
    "generate_two_material_square_mesh",
    "l2_error",
    "load_mesh_and_tags",
    "solve_for_callable_source",
    "solve_for_parameters",
    "sample_random_inclusion_parameters",
]
