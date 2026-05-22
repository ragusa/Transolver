"""Embedding builders for raw Heat2D FOM samples."""

import numpy as np

from heat2d_features import (
    distance_to_outer_boundary,
    nodal_material_2_fraction,
    outer_boundary_mask,
    physical_node_features,
    sample_outer_length,
    signed_distance_to_interface,
    triangle_edge_index,
)


FEATURE_SET_BUILDERS = {}


def register_builder(cls):
    FEATURE_SET_BUILDERS[cls.name] = cls
    return cls


def make_embedding_builder(feature_set):
    try:
        return FEATURE_SET_BUILDERS[feature_set]()
    except KeyError as exc:
        choices = ", ".join(sorted(FEATURE_SET_BUILDERS))
        raise ValueError(f"Unknown Heat2D feature set {feature_set!r}. Choose one of: {choices}") from exc


@register_builder
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

    name = "basic_boundary"
    feature_names = BasicEmbeddingBuilder.feature_names + ("outer_boundary_mask",)

    def _node_features(self, sample, material_2_fraction):
        return np.concatenate(
            [
                super()._node_features(sample, material_2_fraction),
                outer_boundary_mask(sample.coordinates),
            ],
            axis=1,
        )


FEATURE_SET_BUILDERS[BasicWithBoundaryMaskEmbeddingBuilder.name] = BasicWithBoundaryMaskEmbeddingBuilder


@register_builder
class PhysicalEmbeddingBuilder(BasicEmbeddingBuilder):
    """Raw-FOM embedding with derived local physical node fields."""

    name = "physical"
    feature_names = BasicEmbeddingBuilder.feature_names + (
        "kappa_node_arithmetic",
        "kappa_node_harmonic",
        "q_node",
        "outer_boundary_mask",
    )

    def _node_features(self, sample, material_2_fraction):
        kappa_arithmetic, kappa_harmonic, q_node = physical_node_features(sample, material_2_fraction)
        return np.concatenate(
            [
                super()._node_features(sample, material_2_fraction),
                kappa_arithmetic,
                kappa_harmonic,
                q_node,
                outer_boundary_mask(sample.coordinates, outer_length=sample_outer_length(sample)),
            ],
            axis=1,
        )


@register_builder
class GeometryAwareEmbeddingBuilder(BasicEmbeddingBuilder):
    """Raw-FOM embedding with geometric localization fields and no derived physical fields."""

    name = "geometry_aware"
    feature_names = BasicEmbeddingBuilder.feature_names + (
        "outer_boundary_mask",
        "signed_distance_to_interface",
        "distance_to_outer_boundary",
    )

    def _node_features(self, sample, material_2_fraction):
        outer_length = sample_outer_length(sample)
        return np.concatenate(
            [
                super()._node_features(sample, material_2_fraction),
                outer_boundary_mask(sample.coordinates, outer_length=outer_length),
                signed_distance_to_interface(sample),
                distance_to_outer_boundary(sample.coordinates, outer_length=outer_length),
            ],
            axis=1,
        )


@register_builder
class PhysicalPlusEmbeddingBuilder(BasicEmbeddingBuilder):
    """Raw-FOM embedding with physical node fields and geometric localization fields."""

    name = "physical_plus"
    feature_names = BasicEmbeddingBuilder.feature_names + (
        "kappa_node_arithmetic",
        "kappa_node_harmonic",
        "q_node",
        "outer_boundary_mask",
        "signed_distance_to_interface",
        "distance_to_outer_boundary",
    )

    def _node_features(self, sample, material_2_fraction):
        outer_length = sample_outer_length(sample)
        kappa_arithmetic, kappa_harmonic, q_node = physical_node_features(sample, material_2_fraction)
        return np.concatenate(
            [
                super()._node_features(sample, material_2_fraction),
                kappa_arithmetic,
                kappa_harmonic,
                q_node,
                outer_boundary_mask(sample.coordinates, outer_length=outer_length),
                signed_distance_to_interface(sample),
                distance_to_outer_boundary(sample.coordinates, outer_length=outer_length),
            ],
            axis=1,
        )
