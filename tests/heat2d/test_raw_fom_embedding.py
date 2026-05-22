"""Regression tests for raw Heat2D FOM embeddings."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "PDE-Solving-StandardBenchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from heat2d_embeddings import (  # noqa: E402
    BasicEmbeddingBuilder,
    BasicWithBoundaryMaskEmbeddingBuilder,
    GeometryAwareEmbeddingBuilder,
    PhysicalEmbeddingBuilder,
    PhysicalPlusEmbeddingBuilder,
)
from heat2d_features import signed_distance_to_interface  # noqa: E402
from heat2d_raw_sample import Heat2DRawSample  # noqa: E402


def test_raw_fom_basic_builder_matches_legacy_embedding(tmp_path):
    sample_path = _write_raw_fom_sample(tmp_path)

    with np.load(sample_path, allow_pickle=False) as data:
        raw_sample = Heat2DRawSample.from_npz(data, fallback_sample_id=sample_path.stem)
    item = BasicEmbeddingBuilder().build(raw_sample)
    expected = _legacy_raw_fom_item(sample_path, include_boundary_mask=False)

    assert list(BasicEmbeddingBuilder.feature_names) == ["material_2_fraction", "kappa_1", "kappa_2", "q_1", "q_2"]
    _assert_item_matches(item, expected)


def test_raw_fom_boundary_builder_matches_legacy_embedding(tmp_path):
    sample_path = _write_raw_fom_sample(tmp_path)

    with np.load(sample_path, allow_pickle=False) as data:
        raw_sample = Heat2DRawSample.from_npz(data, fallback_sample_id=sample_path.stem)
    item = BasicWithBoundaryMaskEmbeddingBuilder().build(raw_sample)
    expected = _legacy_raw_fom_item(sample_path, include_boundary_mask=True)

    assert list(BasicWithBoundaryMaskEmbeddingBuilder.feature_names) == [
        "material_2_fraction",
        "kappa_1",
        "kappa_2",
        "q_1",
        "q_2",
        "outer_boundary_mask",
    ]
    _assert_item_matches(item, expected)


def test_heat2d_dataset_uses_canonical_basic_builder_names(tmp_path):
    pytest.importorskip("torch")
    from heat2d_dataset import Heat2DDataset

    _write_raw_fom_sample(tmp_path)

    assert Heat2DDataset(tmp_path, split="all").input_feature_names == list(BasicEmbeddingBuilder.feature_names)
    assert Heat2DDataset(tmp_path, split="all").feature_set == "basic"
    assert Heat2DDataset(tmp_path, split="all", include_boundary_mask=True).input_feature_names == list(
        BasicWithBoundaryMaskEmbeddingBuilder.feature_names
    )
    assert Heat2DDataset(tmp_path, split="all", include_boundary_mask=True).feature_set == "basic_boundary"


def test_heat2d_dataset_basic_output_matches_legacy_embedding(tmp_path):
    pytest.importorskip("torch")
    from heat2d_dataset import Heat2DDataset

    sample_path = _write_raw_fom_sample(tmp_path)
    dataset = Heat2DDataset(tmp_path, split="all")

    assert dataset.input_feature_names == list(BasicEmbeddingBuilder.feature_names)
    _assert_dataset_item_matches(dataset[0], _legacy_raw_fom_item(sample_path, include_boundary_mask=False))


def test_heat2d_dataset_boundary_output_matches_legacy_embedding(tmp_path):
    pytest.importorskip("torch")
    from heat2d_dataset import Heat2DDataset

    sample_path = _write_raw_fom_sample(tmp_path)
    dataset = Heat2DDataset(tmp_path, split="all", include_boundary_mask=True)

    assert dataset.input_feature_names == list(BasicWithBoundaryMaskEmbeddingBuilder.feature_names)
    _assert_dataset_item_matches(dataset[0], _legacy_raw_fom_item(sample_path, include_boundary_mask=True))


def test_raw_sample_reader_accepts_scalar_parameter_fields(tmp_path):
    sample_path = _write_raw_fom_sample(tmp_path, scalar_parameters=True)

    with np.load(sample_path, allow_pickle=False) as data:
        sample = Heat2DRawSample.from_npz(data)

    assert sample.kappa_1 == 1.5
    assert sample.kappa_2 == 2.5
    assert sample.q_1 == 3.5
    assert sample.q_2 == 4.5
    np.testing.assert_array_equal(sample.element_kappa, np.array([1.5, 2.5], dtype=np.float32))
    np.testing.assert_array_equal(sample.element_q, np.array([3.5, 4.5], dtype=np.float32))


def test_physical_builder_feature_names_and_values(tmp_path):
    sample = _load_raw_sample(_write_raw_fom_sample(tmp_path))
    item = PhysicalEmbeddingBuilder().build(sample)

    expected_names = [
        "material_2_fraction",
        "kappa_1",
        "kappa_2",
        "q_1",
        "q_2",
        "kappa_node_arithmetic",
        "kappa_node_harmonic",
        "q_node",
        "outer_boundary_mask",
    ]
    f = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float32)
    expected = np.column_stack(
        [
            f,
            np.full(4, 1.5, dtype=np.float32),
            np.full(4, 2.5, dtype=np.float32),
            np.full(4, 3.5, dtype=np.float32),
            np.full(4, 4.5, dtype=np.float32),
            (1.0 - f) * 1.5 + f * 2.5,
            1.0 / ((1.0 - f) / 1.5 + f / 2.5),
            (1.0 - f) * 3.5 + f * 4.5,
            np.ones(4, dtype=np.float32),
        ]
    )

    assert list(PhysicalEmbeddingBuilder.feature_names) == expected_names
    np.testing.assert_allclose(item["node_features"], expected, atol=1e-12, rtol=0.0)


def test_geometry_aware_builder_feature_names_and_finite_distances(tmp_path):
    metadata = {"shape": "disk", "outer_length": 1.0, "center": [0.5, 0.5], "radius": 0.25}
    sample = _load_raw_sample(_write_raw_fom_sample(tmp_path, metadata=metadata))
    item = GeometryAwareEmbeddingBuilder().build(sample)

    assert list(GeometryAwareEmbeddingBuilder.feature_names) == [
        "material_2_fraction",
        "kappa_1",
        "kappa_2",
        "q_1",
        "q_2",
        "outer_boundary_mask",
        "signed_distance_to_interface",
        "distance_to_outer_boundary",
    ]
    assert item["node_features"].shape == (4, 8)
    assert np.all(np.isfinite(item["node_features"]))
    np.testing.assert_array_equal(item["node_features"][:, 5], np.ones(4, dtype=np.float32))
    np.testing.assert_allclose(item["node_features"][:, 7], np.zeros(4, dtype=np.float32), atol=1e-12, rtol=0.0)


def test_physical_plus_builder_feature_names_and_values(tmp_path):
    metadata = {"shape": "disk", "outer_length": 1.0, "center": [0.5, 0.5], "radius": 0.25}
    sample = _load_raw_sample(_write_raw_fom_sample(tmp_path, metadata=metadata))
    item = PhysicalPlusEmbeddingBuilder().build(sample)
    physical = PhysicalEmbeddingBuilder().build(sample)["node_features"]
    geometry = GeometryAwareEmbeddingBuilder().build(sample)["node_features"]

    assert list(PhysicalPlusEmbeddingBuilder.feature_names) == [
        "material_2_fraction",
        "kappa_1",
        "kappa_2",
        "q_1",
        "q_2",
        "kappa_node_arithmetic",
        "kappa_node_harmonic",
        "q_node",
        "outer_boundary_mask",
        "signed_distance_to_interface",
        "distance_to_outer_boundary",
    ]
    np.testing.assert_allclose(item["node_features"][:, :8], physical[:, :8], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(item["node_features"][:, 8], geometry[:, 5], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(item["node_features"][:, 9:], geometry[:, 6:], atol=1e-12, rtol=0.0)


def test_signed_distance_sign_convention_for_disk_square_and_triangle():
    disk = _raw_sample_for_sdf(
        coordinates=np.array([[0.5, 0.5], [0.75, 0.5], [0.9, 0.5]], dtype=np.float32),
        metadata={"shape": "disk", "outer_length": 1.0, "center": [0.5, 0.5], "radius": 0.25},
    )
    square = _raw_sample_for_sdf(
        coordinates=np.array([[0.5, 0.5], [0.7, 0.5], [0.8, 0.5]], dtype=np.float32),
        metadata={"shape": "square", "outer_length": 1.0, "center": [0.5, 0.5], "side_length": 0.4, "rotation": 0.0},
    )
    triangle_side = 0.3
    triangle_top = [0.5, 0.5 + np.sqrt(3.0) * triangle_side / 3.0]
    triangle = _raw_sample_for_sdf(
        coordinates=np.array([[0.5, 0.5], triangle_top, [0.5, 0.8]], dtype=np.float32),
        metadata={
            "shape": "triangle",
            "outer_length": 1.0,
            "center": [0.5, 0.5],
            "side_length": triangle_side,
            "rotation": 0.0,
        },
    )

    for sample in [disk, square, triangle]:
        sdf = signed_distance_to_interface(sample).reshape(-1)
        assert sdf[0] < 0.0
        assert abs(float(sdf[1])) < 1e-6
        assert sdf[2] > 0.0


def test_signed_distance_requires_reconstructable_metadata():
    sample = _raw_sample_for_sdf(
        coordinates=np.array([[0.5, 0.5]], dtype=np.float32),
        metadata={"shape": "disk", "outer_length": 1.0, "center": [0.5, 0.5]},
    )

    with pytest.raises(ValueError, match="radius"):
        signed_distance_to_interface(sample)


def _write_raw_fom_sample(root, scalar_parameters=False, metadata=None):
    geom_dir = root / "geometries" / "geom_00000"
    geom_dir.mkdir(parents=True)
    sample_path = geom_dir / "sample_00000.npz"
    arrays = {
        "coordinates": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "triangles": np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        "material_id": np.array([1, 2], dtype=np.int64),
        "T": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "geometry_metadata_json": np.array(json.dumps(metadata or {"shape": "disk"})),
        "mesh_filename": np.array("mesh.msh"),
        "sample_id": np.array(0, dtype=np.int64),
        "geometry_id": np.array(0, dtype=np.int64),
    }
    if scalar_parameters:
        arrays.update(
            {
                "kappa_1": np.array(1.5, dtype=np.float32),
                "kappa_2": np.array(2.5, dtype=np.float32),
                "q_1": np.array(3.5, dtype=np.float32),
                "q_2": np.array(4.5, dtype=np.float32),
            }
        )
    else:
        arrays.update(
            {
                "param_names": np.array(["kappa_1", "kappa_2", "q_1", "q_2"]),
                "param_values": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
            }
        )
    np.savez_compressed(sample_path, **arrays)
    return sample_path


def _load_raw_sample(sample_path):
    with np.load(sample_path, allow_pickle=False) as data:
        return Heat2DRawSample.from_npz(data, fallback_sample_id=sample_path.stem)


def _raw_sample_for_sdf(coordinates, metadata):
    return Heat2DRawSample(
        coordinates=np.asarray(coordinates, dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        material_id=np.array([2], dtype=np.int64),
        T=np.zeros(coordinates.shape[0], dtype=np.float32),
        kappa_1=1.5,
        kappa_2=2.5,
        q_1=3.5,
        q_2=4.5,
        geometry_metadata_json=json.dumps(metadata),
        mesh_filename=None,
        sample_id=0,
        geometry_id=0,
    )


def _legacy_raw_fom_item(sample_path, include_boundary_mask):
    with np.load(sample_path, allow_pickle=False) as data:
        coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        material_id = np.asarray(data["material_id"], dtype=np.int64)
        target = np.asarray(data["T"], dtype=np.float32).reshape(-1, 1)
        params = {"kappa_1": 1.5, "kappa_2": 2.5, "q_1": 3.5, "q_2": 4.5}
        material_2_fraction = _legacy_nodal_material_2_fraction(coordinates.shape[0], triangles, material_id)
        features = [
            material_2_fraction,
            np.full((coordinates.shape[0], 1), params["kappa_1"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["kappa_2"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["q_1"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["q_2"], dtype=np.float32),
        ]
        if include_boundary_mask:
            features.append(_legacy_outer_boundary_mask(coordinates))
        return {
            "pos": coordinates,
            "node_features": np.concatenate(features, axis=1),
            "target": target,
            "triangles": triangles,
            "element_material_id": material_id,
            "nodal_material_id": (material_2_fraction[:, 0] >= 0.5).astype(np.int64) + 1,
            "edge_index": _legacy_triangle_edge_index(triangles),
        }


def _legacy_nodal_material_2_fraction(n_nodes, triangles, material_id):
    is_material_2 = (material_id == 2).astype(np.float32)
    sums = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for local_node in range(3):
        nodes = triangles[:, local_node]
        np.add.at(sums, nodes, is_material_2)
        np.add.at(counts, nodes, 1.0)
    counts = np.maximum(counts, 1.0)
    return (sums / counts).reshape(n_nodes, 1).astype(np.float32)


def _legacy_outer_boundary_mask(coordinates, tol=1e-6):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mask = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, 1.0, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, 1.0, atol=tol)
    )
    return mask.astype(np.float32).reshape(-1, 1)


def _legacy_triangle_edge_index(triangles):
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


def _assert_item_matches(item, expected):
    np.testing.assert_allclose(item["pos"], expected["pos"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(item["node_features"], expected["node_features"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(item["target"], expected["target"], atol=1e-12, rtol=0.0)
    np.testing.assert_array_equal(item["triangles"], expected["triangles"])
    np.testing.assert_array_equal(item["element_material_id"], expected["element_material_id"])
    np.testing.assert_array_equal(item["nodal_material_id"], expected["nodal_material_id"])
    np.testing.assert_array_equal(item["edge_index"], expected["edge_index"])


def _assert_dataset_item_matches(item, expected):
    _assert_item_matches({name: value.numpy() for name, value in item.items() if name in expected}, expected)
