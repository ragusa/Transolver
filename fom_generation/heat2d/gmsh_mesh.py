"""Gmsh mesh generation for a square with one material inclusion."""

import json
import math
from pathlib import Path

import numpy as np


MATERIAL_1 = 1
MATERIAL_2 = 2
OUTER_BOUNDARY = 101


def generate_two_material_square_mesh(
    output_msh,
    shape="disk",
    outer_length=1.0,
    center=None,
    radius=None,
    side_length=None,
    rotation=0.0,
    mesh_size_background=0.05,
    mesh_size_inclusion=None,
    mesh_size_interface=None,
    seed=None,
):
    """Generate a conforming triangular Gmsh mesh for one inclusion in a square.

    Physical tags are fixed by convention:
    material_1/background = 1, material_2/inclusion = 2, outer_boundary = 101.
    The internal material interface is intentionally left without a boundary
    condition physical group.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError("generate_two_material_square_mesh requires gmsh.") from exc

    shape = shape.lower()
    if shape not in ["disk", "square", "triangle"]:
        raise ValueError("shape must be one of: 'disk', 'square', 'triangle'")

    rng = np.random.default_rng(seed)
    size = _choose_size(shape, outer_length, rng, radius, side_length)
    center = _choose_center(shape, outer_length, rng, center, size, rotation)
    _validate_inclusion_inside(shape, outer_length, center, size, rotation)

    output_msh = Path(output_msh)
    output_msh.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_msh.with_name(output_msh.stem + "_metadata.json")

    mesh_size_interface = (
        mesh_size_interface
        if mesh_size_interface is not None
        else mesh_size_inclusion
        if mesh_size_inclusion is not None
        else 0.5 * mesh_size_background
    )

    gmsh.initialize()
    try:
        gmsh.model.add("heat2d_two_material_square")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        square_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, outer_length, outer_length)
        inclusion_tag = _add_inclusion(gmsh, shape, center, size, rotation)

        # Fragmenting, rather than cutting, keeps both surfaces and forces the
        # generated triangles to conform to the internal material interface.
        gmsh.model.occ.fragment([(2, square_tag)], [(2, inclusion_tag)])
        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
        material_2_surfaces = _classify_inclusion_surfaces(gmsh, surfaces, shape, center, size)
        material_1_surfaces = [tag for tag in surfaces if tag not in material_2_surfaces]
        if not material_1_surfaces or not material_2_surfaces:
            raise RuntimeError("Could not classify both material surfaces after fragmentation.")

        _add_physical_groups(gmsh, material_1_surfaces, material_2_surfaces, outer_length)
        _set_mesh_sizes(gmsh, material_2_surfaces, mesh_size_background, mesh_size_interface)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(output_msh))
    finally:
        gmsh.finalize()

    metadata = {
        "shape": shape,
        "outer_length": outer_length,
        "center": [float(center[0]), float(center[1])],
        "rotation": float(rotation),
        "mesh_size_background": float(mesh_size_background),
        "mesh_size_interface": float(mesh_size_interface),
        "physical_tags": {
            "material_1": MATERIAL_1,
            "material_2": MATERIAL_2,
            "outer_boundary": OUTER_BOUNDARY,
        },
        "seed": seed,
    }
    if shape == "disk":
        metadata["radius"] = float(size)
    else:
        metadata["side_length"] = float(size)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return str(output_msh)


def _choose_size(shape, outer_length, rng, radius, side_length):
    if shape == "disk":
        if radius is not None:
            return float(radius)
        return float(rng.uniform(0.12, 0.25) * outer_length)
    if side_length is not None:
        return float(side_length)
    if shape == "square":
        return float(rng.uniform(0.20, 0.35) * outer_length)
    return float(rng.uniform(0.22, 0.42) * outer_length)


def _choose_center(shape, outer_length, rng, center, size, rotation):
    if center is not None:
        return (float(center[0]), float(center[1]))
    vertices = _shape_vertices(shape, (0.0, 0.0), size, rotation)
    margin = max(max(abs(x), abs(y)) for x, y in vertices) + 0.05 * outer_length
    if shape == "disk":
        margin = size + 0.05 * outer_length
    if 2.0 * margin >= outer_length:
        raise ValueError("Inclusion is too large for the requested outer square.")
    return (
        float(rng.uniform(margin, outer_length - margin)),
        float(rng.uniform(margin, outer_length - margin)),
    )


def _validate_inclusion_inside(shape, outer_length, center, size, rotation):
    tol = 1e-12
    if shape == "disk":
        cx, cy = center
        if cx - size <= tol or cx + size >= outer_length - tol:
            raise ValueError("Disk inclusion must be fully inside the square.")
        if cy - size <= tol or cy + size >= outer_length - tol:
            raise ValueError("Disk inclusion must be fully inside the square.")
        return
    for x, y in _shape_vertices(shape, center, size, rotation):
        if x <= tol or x >= outer_length - tol or y <= tol or y >= outer_length - tol:
            raise ValueError(f"{shape} inclusion must be fully inside the square.")


def _add_inclusion(gmsh, shape, center, size, rotation):
    cx, cy = center
    if shape == "disk":
        return gmsh.model.occ.addDisk(cx, cy, 0.0, size, size)

    vertices = _shape_vertices(shape, center, size, rotation)
    points = [gmsh.model.occ.addPoint(x, y, 0.0) for x, y in vertices]
    lines = []
    for i, point in enumerate(points):
        lines.append(gmsh.model.occ.addLine(point, points[(i + 1) % len(points)]))
    loop = gmsh.model.occ.addCurveLoop(lines)
    return gmsh.model.occ.addPlaneSurface([loop])


def _shape_vertices(shape, center, size, rotation):
    if shape == "disk":
        return [(size, 0.0), (-size, 0.0), (0.0, size), (0.0, -size)]
    if shape == "square":
        half = 0.5 * size
        local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    elif shape == "triangle":
        height = math.sqrt(3.0) * 0.5 * size
        local = [(0.0, 2.0 * height / 3.0), (-0.5 * size, -height / 3.0), (0.5 * size, -height / 3.0)]
    else:
        raise ValueError("Unknown shape.")
    c, s = math.cos(rotation), math.sin(rotation)
    cx, cy = center
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in local]


def _classify_inclusion_surfaces(gmsh, surfaces, shape, center, size):
    expected_area = _inclusion_area(shape, size)
    areas = {tag: gmsh.model.occ.getMass(2, tag) for tag in surfaces}
    candidates = []
    for tag in surfaces:
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        inside = _point_in_inclusion(shape, (x, y), center, size)
        area_error = abs(areas[tag] - expected_area)
        candidates.append((not inside, area_error, tag))

    candidates.sort()
    chosen = candidates[0][2]
    if abs(areas[chosen] - expected_area) > max(1e-8, 0.05 * expected_area):
        raise RuntimeError("Could not identify inclusion surface by area after fragmentation.")
    return [chosen]


def _add_physical_groups(gmsh, material_1_surfaces, material_2_surfaces, outer_length):
    gmsh.model.addPhysicalGroup(2, material_1_surfaces, MATERIAL_1)
    gmsh.model.setPhysicalName(2, MATERIAL_1, "material_1")
    gmsh.model.addPhysicalGroup(2, material_2_surfaces, MATERIAL_2)
    gmsh.model.setPhysicalName(2, MATERIAL_2, "material_2")

    outer_curves = []
    tol = 1e-8 * max(1.0, outer_length)
    for _, tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
        on_left = abs(xmin) < tol and abs(xmax) < tol
        on_right = abs(xmin - outer_length) < tol and abs(xmax - outer_length) < tol
        on_bottom = abs(ymin) < tol and abs(ymax) < tol
        on_top = abs(ymin - outer_length) < tol and abs(ymax - outer_length) < tol
        if on_left or on_right or on_bottom or on_top:
            outer_curves.append(tag)
    if not outer_curves:
        raise RuntimeError("Could not identify outer boundary curves.")

    gmsh.model.addPhysicalGroup(1, outer_curves, OUTER_BOUNDARY)
    gmsh.model.setPhysicalName(1, OUTER_BOUNDARY, "outer_boundary")


def _set_mesh_sizes(gmsh, material_2_surfaces, mesh_size_background, mesh_size_interface):
    all_points = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(all_points, mesh_size_background)

    interface_points = set()
    boundary = gmsh.model.getBoundary([(2, tag) for tag in material_2_surfaces], recursive=True)
    for dim, tag in boundary:
        if dim == 0:
            interface_points.add(tag)
    if interface_points:
        gmsh.model.mesh.setSize([(0, tag) for tag in interface_points], mesh_size_interface)


def _inclusion_area(shape, size):
    if shape == "disk":
        return math.pi * size * size
    if shape == "square":
        return size * size
    return math.sqrt(3.0) * size * size / 4.0


def _point_in_inclusion(shape, point, center, size):
    x, y = point
    cx, cy = center
    if shape == "disk":
        return (x - cx) ** 2 + (y - cy) ** 2 <= size ** 2
    # Area is the primary classifier for polygons. This permissive bounding
    # check only helps prefer the correct surface if more than one area matches.
    return abs(x - cx) <= size and abs(y - cy) <= size
