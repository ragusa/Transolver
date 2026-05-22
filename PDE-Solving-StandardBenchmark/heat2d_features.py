"""Derived node features for raw Heat2D FOM samples."""

import math

import numpy as np


def nodal_material_2_fraction(n_nodes, triangles, material_id):
    is_material_2 = (material_id == 2).astype(np.float32)
    sums = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for local_node in range(3):
        nodes = triangles[:, local_node]
        np.add.at(sums, nodes, is_material_2)
        np.add.at(counts, nodes, 1.0)
    counts = np.maximum(counts, 1.0)
    return (sums / counts).reshape(n_nodes, 1).astype(np.float32)


def outer_boundary_mask(coordinates, outer_length=1.0, tol=1e-6):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mask = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, outer_length, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, outer_length, atol=tol)
    )
    return mask.astype(np.float32).reshape(-1, 1)


def distance_to_outer_boundary(coordinates, outer_length=1.0, normalize=True):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    distance = np.minimum.reduce([x, outer_length - x, y, outer_length - y])
    distance = np.maximum(distance, 0.0)
    if normalize:
        distance = distance / _positive_outer_length(outer_length)
    return distance.astype(np.float32).reshape(-1, 1)


def physical_node_features(sample, material_2_fraction):
    f = material_2_fraction.astype(np.float32)
    kappa_1 = np.float32(sample.kappa_1)
    kappa_2 = np.float32(sample.kappa_2)
    q_1 = np.float32(sample.q_1)
    q_2 = np.float32(sample.q_2)
    kappa_node_arithmetic = (1.0 - f) * kappa_1 + f * kappa_2
    kappa_node_harmonic = 1.0 / ((1.0 - f) / kappa_1 + f / kappa_2)
    q_node = (1.0 - f) * q_1 + f * q_2
    return (
        kappa_node_arithmetic.astype(np.float32),
        kappa_node_harmonic.astype(np.float32),
        q_node.astype(np.float32),
    )


def signed_distance_to_interface(sample, normalize=True):
    metadata = sample.geometry_metadata
    shape = _required_string(metadata, "shape")
    outer_length = _outer_length(metadata)
    center = _required_vector(metadata, "center", 2)
    coordinates = np.asarray(sample.coordinates, dtype=np.float64)

    if shape == "disk":
        radius = _required_float(metadata, "radius")
        sdf = _disk_sdf(coordinates, center, radius)
    elif shape == "square":
        side_length = _required_float(metadata, "side_length")
        rotation = float(metadata.get("rotation", 0.0))
        sdf = _square_sdf(coordinates, center, side_length, rotation)
    elif shape == "triangle":
        side_length = _required_float(metadata, "side_length")
        rotation = float(metadata.get("rotation", 0.0))
        vertices = _triangle_vertices(center, side_length, rotation)
        sdf = _triangle_sdf(coordinates, vertices)
    else:
        raise ValueError(f"Unsupported Heat2D inclusion shape for signed distance: {shape!r}")

    if normalize:
        sdf = sdf / _positive_outer_length(outer_length)
    return sdf.astype(np.float32).reshape(-1, 1)


def triangle_edge_index(triangles):
    if triangles.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    undirected = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    reverse = undirected[:, [1, 0]]
    directed = np.concatenate([undirected, reverse], axis=0)
    directed = np.unique(directed, axis=0)
    return directed.T.astype(np.int64)


def sample_outer_length(sample):
    return _outer_length(sample.geometry_metadata)


def _disk_sdf(coordinates, center, radius):
    delta = coordinates - np.asarray(center, dtype=np.float64).reshape(1, 2)
    return np.linalg.norm(delta, axis=1) - radius


def _square_sdf(coordinates, center, side_length, rotation):
    local = _to_local_frame(coordinates, center, rotation)
    half = 0.5 * side_length
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def _triangle_sdf(coordinates, vertices):
    distances = np.stack(
        [
            _distance_to_segment(coordinates, vertices[0], vertices[1]),
            _distance_to_segment(coordinates, vertices[1], vertices[2]),
            _distance_to_segment(coordinates, vertices[2], vertices[0]),
        ],
        axis=1,
    )
    unsigned = distances.min(axis=1)
    inside = _points_in_triangle(coordinates, vertices)
    return np.where(inside, -unsigned, unsigned)


def _distance_to_segment(points, start, end):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    edge = end - start
    denom = np.dot(edge, edge)
    if denom <= 0.0:
        return np.linalg.norm(points - start.reshape(1, 2), axis=1)
    t = np.sum((points - start.reshape(1, 2)) * edge.reshape(1, 2), axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = start.reshape(1, 2) + t.reshape(-1, 1) * edge.reshape(1, 2)
    return np.linalg.norm(points - closest, axis=1)


def _points_in_triangle(points, vertices, tol=1e-12):
    v0, v1, v2 = vertices
    area = _cross_2d(v1 - v0, v2 - v0)
    if abs(area) <= tol:
        raise ValueError("Triangle metadata produce a degenerate inclusion.")
    c0 = _cross_2d(v1 - v0, points - v0.reshape(1, 2))
    c1 = _cross_2d(v2 - v1, points - v1.reshape(1, 2))
    c2 = _cross_2d(v0 - v2, points - v2.reshape(1, 2))
    if area > 0.0:
        return (c0 >= -tol) & (c1 >= -tol) & (c2 >= -tol)
    return (c0 <= tol) & (c1 <= tol) & (c2 <= tol)


def _cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _triangle_vertices(center, side_length, rotation):
    height = math.sqrt(3.0) * 0.5 * side_length
    local = np.array(
        [
            [0.0, 2.0 * height / 3.0],
            [-0.5 * side_length, -height / 3.0],
            [0.5 * side_length, -height / 3.0],
        ],
        dtype=np.float64,
    )
    return _from_local_frame(local, center, rotation)


def _to_local_frame(coordinates, center, rotation):
    shifted = coordinates - np.asarray(center, dtype=np.float64).reshape(1, 2)
    c, s = math.cos(rotation), math.sin(rotation)
    return np.column_stack((c * shifted[:, 0] + s * shifted[:, 1], -s * shifted[:, 0] + c * shifted[:, 1]))


def _from_local_frame(local, center, rotation):
    c, s = math.cos(rotation), math.sin(rotation)
    center = np.asarray(center, dtype=np.float64).reshape(1, 2)
    return np.column_stack((c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1])) + center


def _outer_length(metadata):
    return float(metadata.get("outer_length", 1.0))


def _positive_outer_length(outer_length):
    outer_length = float(outer_length)
    if outer_length <= 0.0:
        raise ValueError(f"outer_length must be positive, got {outer_length!r}")
    return outer_length


def _required_float(metadata, name):
    if name not in metadata:
        raise ValueError(_missing_metadata_message(name))
    value = metadata[name]
    if value is None:
        raise ValueError(_missing_metadata_message(name))
    return float(value)


def _required_string(metadata, name):
    if name not in metadata:
        raise ValueError(_missing_metadata_message(name))
    return str(metadata[name]).lower()


def _required_vector(metadata, name, length):
    if name not in metadata:
        raise ValueError(_missing_metadata_message(name))
    value = np.asarray(metadata[name], dtype=np.float64)
    if value.shape != (length,):
        raise ValueError(
            f"Heat2D geometry_metadata_json field {name!r} must have shape ({length},); "
            "the smallest generator metadata addition is to store the inclusion center as [cx, cy]."
        )
    return value


def _missing_metadata_message(name):
    return (
        f"Heat2D geometry_metadata_json is missing {name!r}, which is required for analytic signed distance. "
        "The smallest generator metadata addition is to store shape, outer_length, center, rotation, and "
        "radius for disks or side_length for squares/triangles."
    )
