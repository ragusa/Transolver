"""Smoke-test loader for the heat2d pilot dataset."""

import argparse
import json
from pathlib import Path

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


def run_smoke_test(dataset_dir="fom_generation/data/heat2d_pilot"):
    """Load the pilot manifest and verify all samples."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    seen_shapes = set()
    print("Heat2D pilot loader smoke test")
    print(f"dataset: {dataset_dir}")
    print(f"schema: {manifest.get('schema_version')}")
    print(f"samples: {len(manifest['samples'])}")
    print(f"splits: {manifest['splits']}")

    for sample in manifest["samples"]:
        sample_path = dataset_dir / sample["file"]
        with np.load(sample_path, allow_pickle=False) as data:
            _check_required_fields(data, sample_path)
            shape = str(data["shape_type"])
            seen_shapes.add(shape)
            _check_shapes_and_values(data, sample_path)
            print(f"\n{sample['sample_name']} ({shape})")
            for field in REQUIRED_FIELDS:
                value = data[field]
                print(f"  {field}: shape={value.shape}, dtype={value.dtype}")

    missing = {"disk", "square", "triangle"} - seen_shapes
    if missing:
        raise AssertionError(f"Pilot dataset is missing shapes: {sorted(missing)}")

    print("\nPilot loader smoke test passed")


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-test the heat2d pilot dataset loader.")
    parser.add_argument("--dataset", default="fom_generation/data/heat2d_pilot")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_smoke_test(args.dataset)


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


if __name__ == "__main__":
    main()
