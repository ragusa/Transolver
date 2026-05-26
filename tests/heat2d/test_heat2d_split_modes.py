"""Tests for Heat2D sample-level and geometry-level raw-FOM splits."""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "PDE-Solving-StandardBenchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from exp_heat2d import build_parser  # noqa: E402
from heat2d_dataset import (  # noqa: E402
    Heat2DDataset,
    _assign_sample_level_splits,
    assign_heat2d_splits,
    heat2d_geometry_key,
)


@pytest.fixture
def workspace_tmp():
    path = REPO_ROOT / "pytest_manual_tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_split_mode_default_is_sample():
    args = build_parser().parse_args([])
    explicit = build_parser().parse_args(["--split-mode", "sample"])

    assert args.split_mode == "sample"
    assert explicit.split_mode == "sample"


def test_sample_split_matches_existing_assignment(workspace_tmp):
    samples = _synthetic_sample_records(workspace_tmp, n=10)

    old = _assign_sample_level_splits(samples, split_seed=7, split_fractions=(0.8, 0.1, 0.1))
    new = assign_heat2d_splits(samples, split_seed=7, split_fractions=(0.8, 0.1, 0.1), split_mode="sample")

    assert [(row["sample_name"], row["split"]) for row in new] == [
        (row["sample_name"], row["split"]) for row in old
    ]


def test_geometry_split_keeps_each_geometry_in_one_split(workspace_tmp):
    root = _write_raw_fom_dataset(workspace_tmp / "root", n_geometries=6, samples_per_geometry=2)

    split_keys = {}
    for split in ("train", "val", "test"):
        dataset = Heat2DDataset(root, split=split, split_mode="geometry")
        split_keys[split] = {sample["geometry_key"] for sample in dataset.samples}

    assert split_keys["train"].isdisjoint(split_keys["val"])
    assert split_keys["train"].isdisjoint(split_keys["test"])
    assert split_keys["val"].isdisjoint(split_keys["test"])


def test_geometry_key_distinguishes_same_numeric_geometry_across_roots(workspace_tmp):
    root_a = _write_raw_fom_dataset(workspace_tmp / "root_a", n_geometries=1, samples_per_geometry=1, shape="disk")
    root_b = _write_raw_fom_dataset(workspace_tmp / "root_b", n_geometries=1, samples_per_geometry=1, shape="square")

    sample_a = root_a / "geometries" / "geom_00000" / "sample_00000.npz"
    sample_b = root_b / "geometries" / "geom_00000" / "sample_00000.npz"
    assert heat2d_geometry_key(root_a, sample_a) != heat2d_geometry_key(root_b, sample_b)

    dataset = Heat2DDataset([root_a, root_b], split="all", split_mode="geometry")
    assert dataset.split_diagnostics["geometry_group_count"] == 2
    assert len({sample["geometry_key"] for sample in dataset.samples}) == 2


@pytest.mark.parametrize("feature_set", ["basic", "physical_plus"])
def test_geometry_split_supports_raw_fom_feature_sets(workspace_tmp, feature_set):
    pytest.importorskip("torch")
    root = _write_raw_fom_dataset(workspace_tmp / "root", n_geometries=4, samples_per_geometry=1)

    dataset = Heat2DDataset(root, split="train", split_mode="geometry", feature_set=feature_set)
    item = dataset[0]

    assert dataset.feature_set == feature_set
    assert item["node_features"].shape[1] == len(dataset.input_feature_names)


def test_tiny_geometry_split_train_completes(workspace_tmp):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    root = _write_raw_fom_dataset(workspace_tmp / "data", n_geometries=4, samples_per_geometry=1)
    out = workspace_tmp / "run"

    result = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_DIR / "exp_heat2d.py"),
            "--mode",
            "train",
            "--data_path",
            str(root),
            "--split-mode",
            "geometry",
            "--feature-set",
            "basic",
            "--epochs",
            "1",
            "--eval-every",
            "1",
            "--n-hidden",
            "8",
            "--n-layers",
            "1",
            "--n-heads",
            "1",
            "--slice_num",
            "2",
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-3000:]
    summary = json.loads((out / "split_summary.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    run_summary = json.loads((out / "run_summary.json").read_text())

    assert summary["split_mode"] == "geometry"
    assert summary["geometry_groups_disjoint"] is True
    assert metrics["split_mode"] == "geometry"
    assert run_summary["split_mode"] == "geometry"


def _synthetic_sample_records(root, n):
    samples = []
    for index in range(n):
        geom = root / "geometries" / f"geom_{index:05d}"
        path = geom / "sample_00000.npz"
        samples.append(
            {
                "schema": "fom",
                "root": root,
                "path": path,
                "sample_name": f"sample_{index}",
                "geometry_key": f"{root.resolve()}::geometries/geom_{index:05d}",
            }
        )
    return samples


def _write_raw_fom_dataset(root, n_geometries, samples_per_geometry, shape="disk"):
    root.mkdir(parents=True, exist_ok=True)
    for geometry_id in range(n_geometries):
        geom_dir = root / "geometries" / f"geom_{geometry_id:05d}"
        geom_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in range(samples_per_geometry):
            _write_raw_fom_sample(geom_dir / f"sample_{sample_id:05d}.npz", geometry_id, sample_id, shape)
    return root


def _write_raw_fom_sample(path, geometry_id, sample_id, shape):
    metadata = {
        "shape": shape,
        "outer_length": 1.0,
        "center": [0.5, 0.5],
        "radius": 0.25,
        "side_length": 0.4,
        "rotation": 0.0,
    }
    np.savez_compressed(
        path,
        coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        material_id=np.array([1, 2], dtype=np.int64),
        T=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64) + float(sample_id),
        geometry_metadata_json=np.array(json.dumps(metadata)),
        mesh_filename=np.array("mesh.msh"),
        sample_id=np.array(sample_id, dtype=np.int64),
        geometry_id=np.array(geometry_id, dtype=np.int64),
        param_names=np.array(["kappa_1", "kappa_2", "q_1", "q_2"]),
        param_values=np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
    )
