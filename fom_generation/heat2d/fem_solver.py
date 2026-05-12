"""scikit-fem solve utilities for the two-material 2D heat equation."""

import json
import warnings
from pathlib import Path

import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm
from skfem.helpers import dot, grad


MATERIAL_1 = 1
MATERIAL_2 = 2
OUTER_BOUNDARY = 101


class MeshData:
    """Small container for mesh arrays and physical tags."""

    def __init__(self, mesh, coordinates, triangles, material_id, outer_boundary_nodes, msh_path):
        self.mesh = mesh
        self.coordinates = coordinates
        self.triangles = triangles
        self.material_id = material_id
        self.outer_boundary_nodes = outer_boundary_nodes
        self.msh_path = str(msh_path)


class AffineComponents:
    """Affine operators for repeated solves on one fixed mesh."""

    def __init__(self, K1, K2, F1, F2, basis, dirichlet_dofs, mesh_data, mesh_info):
        self.K1 = K1
        self.K2 = K2
        self.F1 = F1
        self.F2 = F2
        self.basis = basis
        self.dirichlet_dofs = dirichlet_dofs
        self.mesh_data = mesh_data
        self.mesh_info = mesh_info
        self.last_residual_norm = None
        self.last_relative_residual_norm = None


def load_mesh_and_tags(msh_path):
    """Load a Gmsh mesh and preserve physical tags.

    meshio is used directly because scikit-fem's convenience loaders may not
    expose the Gmsh physical IDs for every .msh version. We then build a
    MeshTri manually from the same coordinates and triangular connectivity.
    """
    msh_path = Path(msh_path)
    raw = meshio.read(msh_path)
    points = np.asarray(raw.points[:, :2], dtype=float)

    triangles = []
    triangle_tags = []
    outer_lines = []
    if "gmsh:physical" not in raw.cell_data:
        raise ValueError("Mesh is missing gmsh:physical cell data.")

    for block, physical in zip(raw.cells, raw.cell_data["gmsh:physical"]):
        if block.type == "triangle":
            triangles.append(np.asarray(block.data, dtype=np.int64))
            triangle_tags.append(np.asarray(physical, dtype=np.int64))
        elif block.type == "line":
            physical = np.asarray(physical, dtype=np.int64)
            lines = np.asarray(block.data, dtype=np.int64)
            if np.any(physical == OUTER_BOUNDARY):
                outer_lines.append(lines[physical == OUTER_BOUNDARY])

    if not triangles:
        raise ValueError("Mesh contains no triangular cells.")
    if not outer_lines:
        raise ValueError("Mesh contains no outer_boundary physical line cells.")

    triangles = np.vstack(triangles)
    material_id = np.concatenate(triangle_tags)
    outer_boundary_nodes = np.unique(np.vstack(outer_lines).ravel())

    if not np.all(np.isin([MATERIAL_1, MATERIAL_2], np.unique(material_id))):
        raise ValueError("Material tags 1 and 2 must both be present.")
    if outer_boundary_nodes.size == 0:
        raise ValueError("No Dirichlet boundary nodes were found.")

    mesh = MeshTri(points.T, triangles.T)
    return MeshData(mesh, points, triangles, material_id, outer_boundary_nodes, msh_path)


@BilinearForm
def _laplace(u, v, w):
    return dot(grad(u), grad(v))


@LinearForm
def _unit_load(v, w):
    return v


@BilinearForm
def _mass(u, v, w):
    return u * v


def assemble_affine_components(mesh_data):
    """Assemble K1, K2, F1, F2 for piecewise constant materials and sources."""
    material_id = np.asarray(mesh_data.material_id)
    elements_1 = np.nonzero(material_id == MATERIAL_1)[0]
    elements_2 = np.nonzero(material_id == MATERIAL_2)[0]
    if elements_1.size == 0 or elements_2.size == 0:
        raise ValueError("Both material regions must contain at least one element.")

    basis = Basis(mesh_data.mesh, ElementTriP1())
    basis_1 = basis.with_elements(elements_1)
    basis_2 = basis.with_elements(elements_2)

    K1 = csr_matrix(asm(_laplace, basis_1))
    K2 = csr_matrix(asm(_laplace, basis_2))
    F1 = np.asarray(asm(_unit_load, basis_1)).ravel()
    F2 = np.asarray(asm(_unit_load, basis_2)).ravel()

    dirichlet_dofs = np.asarray(mesh_data.outer_boundary_nodes, dtype=np.int64)
    if dirichlet_dofs.size == 0:
        raise ValueError("Dirichlet dofs are empty.")
    if dirichlet_dofs.max() >= basis.N:
        raise ValueError("Boundary node index exceeds basis size.")

    mesh_info = {
        "n_nodes": int(mesh_data.coordinates.shape[0]),
        "n_elements": int(mesh_data.triangles.shape[0]),
        "n_material_1_elements": int(elements_1.size),
        "n_material_2_elements": int(elements_2.size),
        "n_dirichlet_dofs": int(dirichlet_dofs.size),
    }
    return AffineComponents(K1, K2, F1, F2, basis, dirichlet_dofs, mesh_data, mesh_info)


def assemble_callable_source_vector(basis, source, elements=None):
    """Assemble a load vector for a callable source q(x, y).

    The callable must accept vectorized x and y arrays from scikit-fem
    quadrature points and return values with the same shape.
    """
    local_basis = basis.with_elements(elements) if elements is not None else basis

    @LinearForm
    def _source_load(v, w):
        return source(w.x[0], w.x[1]) * v

    return np.asarray(asm(_source_load, local_basis)).ravel()


def solve_for_parameters(affine_components, kappa_1, kappa_2, q_1, q_2):
    """Solve the homogeneous-Dirichlet heat problem for one parameter set."""
    if kappa_1 <= 0.0 or kappa_2 <= 0.0:
        raise ValueError("Conductivities kappa_1 and kappa_2 must be positive.")

    A = kappa_1 * affine_components.K1 + kappa_2 * affine_components.K2
    b = q_1 * affine_components.F1 + q_2 * affine_components.F2
    T = _solve_homogeneous_dirichlet(A, b, affine_components.dirichlet_dofs)

    _store_residual_info(affine_components, A, b, T)
    if T.min() < -1e-8 and q_1 >= 0.0 and q_2 >= 0.0:
        warnings.warn(f"Solution has a strongly negative minimum: {T.min():.3e}")
    return T


def solve_for_callable_source(affine_components, kappa_1, kappa_2, source):
    """Solve with piecewise-constant kappa and a callable volumetric source."""
    if kappa_1 <= 0.0 or kappa_2 <= 0.0:
        raise ValueError("Conductivities kappa_1 and kappa_2 must be positive.")

    A = kappa_1 * affine_components.K1 + kappa_2 * affine_components.K2
    b = assemble_callable_source_vector(affine_components.basis, source)
    T = _solve_homogeneous_dirichlet(A, b, affine_components.dirichlet_dofs)

    _store_residual_info(affine_components, A, b, T)
    return T


def l2_error(basis, numerical, exact):
    """Compute absolute and relative L2-like errors using the P1 mass matrix."""
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    if numerical.shape != exact.shape:
        raise ValueError("numerical and exact arrays must have the same shape.")
    mass = csr_matrix(asm(_mass, basis))
    error = numerical - exact
    absolute = float(np.sqrt(max(error @ (mass @ error), 0.0)))
    exact_norm = float(np.sqrt(max(exact @ (mass @ exact), 0.0)))
    relative = absolute / max(exact_norm, 1e-14)
    return absolute, relative


def save_affine_info(affine_components, path):
    """Save lightweight assembly information for a generated geometry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = dict(affine_components.mesh_info)
    info["msh_path"] = affine_components.mesh_data.msh_path
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def _solve_homogeneous_dirichlet(A, b, dirichlet_dofs):
    n = A.shape[0]
    free = _free_dofs(n, dirichlet_dofs)
    if free.size == 0:
        raise ValueError("No free dofs remain after applying Dirichlet conditions.")
    A_ff = A[free][:, free].tocsr()
    b_f = b[free]
    T = np.zeros(n, dtype=float)
    T[free] = spsolve(A_ff, b_f)
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Linear solve produced non-finite values.")
    return T


def _store_residual_info(affine_components, A, b, T):
    residual = A @ T - b
    free = _free_dofs(A.shape[0], affine_components.dirichlet_dofs)
    residual_norm = float(np.linalg.norm(residual[free]))
    b_norm = float(np.linalg.norm(b[free]))
    relative = residual_norm / max(1.0, b_norm)
    affine_components.last_residual_norm = residual_norm
    affine_components.last_relative_residual_norm = relative
    if relative > 1e-8:
        warnings.warn(f"Large free-dof residual after solve: {relative:.3e}")


def _free_dofs(n, dirichlet_dofs):
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(dirichlet_dofs, dtype=np.int64)] = False
    return np.nonzero(mask)[0]
