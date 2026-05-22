"""Embedding builders for raw Heat2D FOM samples."""

import numpy as np


class BasicEmbeddingBuilder:
    """Current raw-FOM node embedding without a boundary mask."""

    name = "basic"
    feature_names = ("material_2_fraction", "kappa_1", "kappa_2", "q_1", "q_2")

    def build(self, sample):
        material_2_fraction = nodal_material_2_fraction(
            sample.coordinates.shape[0],
            sample.triangles,
            sample.material_id,
        )
        node_features = self._node_features(sample, material_2_fraction)
        return {
            "pos": sample.coordinates,
            "node_features": node_features,
            "target": sample.T.reshape(-1, 1),
            "triangles": sample.triangles,
            "element_material_id": sample.material_id,
            "nodal_material_id": (material_2_fraction[:, 0] >= 0.5).astype(np.int64) + 1,
            "edge_index": triangle_edge_index(sample.triangles),
        }

    def _node_features(self, sample, material_2_fraction):
        return np.concatenate(
            [
                material_2_fraction,
                np.full((sample.coordinates.shape[0], 1), sample.kappa_1, dtype=np.float32),
                np.full((sample.coordinates.shape[0], 1), sample.kappa_2, dtype=np.float32),
                np.full((sample.coordinates.shape[0], 1), sample.q_1, dtype=np.float32),
                np.full((sample.coordinates.shape[0], 1), sample.q_2, dtype=np.float32),
            ],
            axis=1,
        )


class BasicWithBoundaryMaskEmbeddingBuilder(BasicEmbeddingBuilder):
    """Current raw-FOM node embedding with the boundary mask appended."""

    name = "basic_with_boundary_mask"
    feature_names = BasicEmbeddingBuilder.feature_names + ("outer_boundary_mask",)

    def _node_features(self, sample, material_2_fraction):
        return np.concatenate(
            [
                super()._node_features(sample, material_2_fraction),
                outer_boundary_mask(sample.coordinates),
            ],
            axis=1,
        )


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


def outer_boundary_mask(coordinates, tol=1e-6):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mask = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, 1.0, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, 1.0, atol=tol)
    )
    return mask.astype(np.float32).reshape(-1, 1)


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
