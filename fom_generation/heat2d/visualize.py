"""Small matplotlib helpers for checking generated heat2d meshes and solutions."""

import matplotlib.tri as mtri
import numpy as np


def plot_mesh_with_materials(mesh_data, ax=None):
    """Plot triangular elements colored by material ID."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    triang = mtri.Triangulation(
        mesh_data.coordinates[:, 0],
        mesh_data.coordinates[:, 1],
        mesh_data.triangles,
    )
    values = np.asarray(mesh_data.material_id)
    patch = ax.tripcolor(triang, facecolors=values, edgecolors="k", linewidth=0.25, cmap="Set2")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Material tags")
    return patch


def plot_solution(mesh_data, T, ax=None):
    """Plot a nodal P1 temperature solution."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    triang = mtri.Triangulation(
        mesh_data.coordinates[:, 0],
        mesh_data.coordinates[:, 1],
        mesh_data.triangles,
    )
    contour = ax.tricontourf(triang, T, levels=30, cmap="inferno")
    ax.triplot(triang, color="k", linewidth=0.15, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Temperature")
    return contour
