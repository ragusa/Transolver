"""Utilities for 2D heat-conduction full-order model generation."""

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


def __getattr__(name):
    if name in {
        "assemble_affine_components",
        "assemble_callable_source_vector",
        "l2_error",
        "load_mesh_and_tags",
        "solve_for_callable_source",
        "solve_for_parameters",
    }:
        from . import fem_solver

        return getattr(fem_solver, name)
    if name in {"generate_two_material_square_mesh", "sample_random_inclusion_parameters"}:
        from . import gmsh_mesh

        return getattr(gmsh_mesh, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
