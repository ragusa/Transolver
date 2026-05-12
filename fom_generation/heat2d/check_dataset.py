"""Integrity checks and summary reporting for Heat2D NPZ datasets."""

import argparse
import json
from collections import Counter, defaultdict
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
    "shape_type_id",
    "geometry_metadata_json",
    "boundary_condition",
    "pde_sign_convention",
]


def check_dataset(dataset_dir, summary_path=None, write_summary=True):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    shape_counts = Counter()
    split_counts = Counter()
    split_shape_counts = defaultdict(Counter)
    node_counts = []
    element_counts = []
    parameter_values = {name: [] for name in ["kappa_1", "kappa_2", "q_1", "q_2"]}

    for sample in manifest["samples"]:
        sample_path = dataset_dir / sample["file"]
        split = sample.get("split", "unknown")
        try:
            with np.load(sample_path, allow_pickle=False) as data:
                _check_sample(data, sample_path)
                shape = str(data["shape_type"])
                shape_counts[shape] += 1
                split_counts[split] += 1
                split_shape_counts[split][shape] += 1
                node_counts.append(int(data["pos"].shape[0]))
                element_counts.append(int(data["triangles"].shape[0]))
                for name in parameter_values:
                    parameter_values[name].append(float(data[name]))
        except Exception as exc:  # noqa: BLE001 - report all integrity failures together.
            errors.append(f"{sample_path}: {exc}")

    if errors:
        joined = "\n".join(errors)
        raise AssertionError(f"Heat2D dataset integrity check failed:\n{joined}")

    summary = {
        "output_directory": str(dataset_dir),
        "num_samples": len(manifest["samples"]),
        "shape_counts": dict(sorted(shape_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "shape_counts_per_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_shape_counts.items())
        },
        "node_count": _min_mean_max(node_counts),
        "element_count": _min_mean_max(element_counts),
        "parameter_ranges": {
            name: _min_mean_max(values) for name, values in parameter_values.items()
        },
    }

    if write_summary:
        if summary_path is None:
            summary_path = dataset_dir / "dataset_summary.json"
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    return summary


def _check_sample(data, sample_path):
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        raise AssertionError(f"missing fields {missing}")

    pos = data["pos"]
    node_features = data["node_features"]
    target = data["target"]
    triangles = data["triangles"]
    element_material_id = data["element_material_id"]
    edge_index = data["edge_index"]
    shape_type = str(data["shape_type"])
    shape_type_id = int(data["shape_type_id"])

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise AssertionError(f"pos must have shape (N, 2), got {pos.shape}")
    if node_features.shape != (pos.shape[0], 8):
        raise AssertionError(f"node_features must have shape (N, 8), got {node_features.shape}")
    if target.shape != (pos.shape[0], 1):
        raise AssertionError(f"target must have shape (N, 1), got {target.shape}")
    if not np.all(np.isfinite(pos)):
        raise AssertionError("pos contains non-finite values")
    if not np.all(np.isfinite(node_features)):
        raise AssertionError("node_features contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise AssertionError("target contains non-finite values")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise AssertionError(f"triangles must have shape (E, 3), got {triangles.shape}")
    if element_material_id.shape != (triangles.shape[0],):
        raise AssertionError("element_material_id must align with triangles")
    if not np.all(np.isin([1, 2], np.unique(element_material_id))):
        raise AssertionError("both material IDs 1 and 2 must be present")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= pos.shape[0]):
        raise AssertionError("triangles contain node indices outside pos")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    ):
        raise AssertionError("triangles contain repeated node indices")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise AssertionError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= pos.shape[0]):
        raise AssertionError("edge_index contains node indices outside pos")
    expected_shape_ids = {"disk": 0, "square": 1, "triangle": 2}
    if shape_type not in expected_shape_ids:
        raise AssertionError(f"unknown shape_type {shape_type!r}")
    if shape_type_id != expected_shape_ids[shape_type]:
        raise AssertionError(f"shape_type_id {shape_type_id} does not match {shape_type}")


def _min_mean_max(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Check and summarize a Heat2D dataset.")
    parser.add_argument("--dataset", default="fom_generation/data/heat2d_moderate_balanced")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--no-write-summary", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = check_dataset(
        args.dataset,
        summary_path=args.summary_path,
        write_summary=not args.no_write_summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
