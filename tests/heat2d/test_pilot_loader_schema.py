"""Tests for the lightweight Heat2D pilot sample schema."""

import json

import numpy as np


REQUIRED_FIELDS = [
    "pos",
    "node_features",
    "input_feature_names",
    "target",
    "coordinates",
    "triangles",
    "element_material_id",
    "nodal_material_id",
    "edge_index",
    "kappa_1",
    "kappa_2",
    "q_1",
    "q_2",
    "shape_type",
    "geometry_metadata_json",
    "boundary_condition",
    "pde_sign_convention",
]


def test_minimal_pilot_manifest_and_samples_are_loadable(tmp_path):
    samples = []
    for sample_id, shape in enumerate(["disk", "square", "triangle"]):
        sample_name = f"sample_{sample_id:05d}_{shape}"
        sample_dir = tmp_path / "samples" / sample_name
        sample_dir.mkdir(parents=True)
        sample_path = sample_dir / "sample.npz"
        _write_sample(sample_path, sample_id, shape)
        samples.append(
            {
                "sample_id": sample_id,
                "sample_name": sample_name,
                "file": f"samples/{sample_name}/sample.npz",
                "shape": shape,
                "split": "train",
            }
        )

    manifest = {
        "schema_version": "heat2d_pilot_v1",
        "samples": samples,
        "splits": {"train": [sample["sample_name"] for sample in samples]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    seen_shapes = set()
    for sample in loaded_manifest["samples"]:
        sample_path = tmp_path / sample["file"]
        with np.load(sample_path, allow_pickle=False) as data:
            _check_required_fields(data, sample_path)
            seen_shapes.add(str(data["shape_type"]))
            _check_shapes_and_values(data, sample_path)

    assert seen_shapes == {"disk", "square", "triangle"}


def _write_sample(path, sample_id, shape):
    pos = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    np.savez_compressed(
        path,
        pos=pos,
        node_features=np.ones((4, 8), dtype=np.float32),
        input_feature_names=np.array(
            ["kappa_node", "q_node", "material_2_fraction", "outer_boundary_mask", "kappa_1", "kappa_2", "q_1", "q_2"]
        ),
        target=np.arange(4, dtype=np.float32)[:, None],
        coordinates=pos,
        triangles=triangles,
        element_material_id=np.array([1, 2], dtype=np.int64),
        nodal_material_id=np.array([1, 1, 2, 2], dtype=np.int64),
        edge_index=np.array([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=np.int64),
        kappa_1=np.array(1.0, dtype=np.float32),
        kappa_2=np.array(2.0, dtype=np.float32),
        q_1=np.array(3.0, dtype=np.float32),
        q_2=np.array(4.0, dtype=np.float32),
        shape_type=np.array(shape),
        shape_type_id=np.array(sample_id, dtype=np.int64),
        geometry_metadata_json=np.array("{}"),
        boundary_condition=np.array("T=0 on outer_boundary"),
        pde_sign_convention=np.array("-div(kappa grad T)=q"),
    )


def _check_required_fields(data, path):
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        raise AssertionError(f"{path}: missing fields {missing}")


def _check_shapes_and_values(data, path):
    pos = data["pos"]
    node_features = data["node_features"]
    target = data["target"]
    triangles = data["triangles"]
    material = data["element_material_id"]
    edge_index = data["edge_index"]

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise AssertionError(f"{path}: pos must have shape (n_nodes, 2)")
    if node_features.ndim != 2 or node_features.shape[0] != pos.shape[0]:
        raise AssertionError(f"{path}: node_features must align with pos")
    if target.shape != (pos.shape[0], 1):
        raise AssertionError(f"{path}: target must have shape (n_nodes, 1)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise AssertionError(f"{path}: triangles must have shape (n_elements, 3)")
    if material.shape != (triangles.shape[0],):
        raise AssertionError(f"{path}: element_material_id must align with triangles")
    if not np.all(np.isin([1, 2], np.unique(material))):
        raise AssertionError(f"{path}: both material IDs 1 and 2 must be present")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise AssertionError(f"{path}: edge_index must have shape (2, n_edges)")
    for field in ["pos", "node_features", "target"]:
        if not np.all(np.isfinite(data[field])):
            raise AssertionError(f"{path}: {field} contains non-finite values")
