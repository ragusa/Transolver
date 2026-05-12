import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_heat2d_dataset_dir(dataset_dir):
    """Resolve a heat2d dataset path from repo root or benchmark directory."""
    path = Path(dataset_dir)
    if path.exists():
        return path

    benchmark_dir = Path(__file__).resolve().parent
    repo_root = benchmark_dir.parent
    candidates = [
        repo_root / dataset_dir,
        benchmark_dir / dataset_dir,
        benchmark_dir / ".." / dataset_dir,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find Heat2D dataset directory: {dataset_dir}")


class Heat2DDataset(Dataset):
    """Minimal loader for one-sample-per-NPZ Heat2D FOM data."""

    def __init__(self, dataset_dir="fom_generation/data/heat2d_pilot", split="train", sample_names=None):
        self.dataset_dir = resolve_heat2d_dataset_dir(dataset_dir)
        self.manifest_path = self.dataset_dir / "manifest.json"
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        samples = list(self.manifest["samples"])
        if split != "all":
            samples = [sample for sample in samples if sample.get("split") == split]
        if sample_names is not None:
            wanted = set(sample_names)
            samples = [sample for sample in samples if sample["sample_name"] in wanted]

        if not samples:
            raise ValueError(f"No Heat2D samples found for split={split!r}")

        self.split = split
        self.samples = samples
        self.input_feature_names = list(self.manifest.get("arrays", {}).get("input_feature_names", []))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = self.dataset_dir / sample["file"]
        with np.load(sample_path, allow_pickle=False) as data:
            item = {
                "pos": torch.as_tensor(data["pos"], dtype=torch.float32),
                "node_features": torch.as_tensor(data["node_features"], dtype=torch.float32),
                "target": torch.as_tensor(data["target"], dtype=torch.float32),
                "triangles": torch.as_tensor(data["triangles"], dtype=torch.long),
                "element_material_id": torch.as_tensor(data["element_material_id"], dtype=torch.long),
                "nodal_material_id": torch.as_tensor(data["nodal_material_id"], dtype=torch.long),
                "edge_index": torch.as_tensor(data["edge_index"], dtype=torch.long),
                "shape_type": str(data["shape_type"]),
                "shape_type_id": int(data["shape_type_id"]) if "shape_type_id" in data.files else None,
                "geometry_metadata_json": str(data["geometry_metadata_json"]),
                "sample_name": sample["sample_name"],
                "sample_id": int(data["sample_id"]) if "sample_id" in data.files else int(sample.get("sample_id", idx)),
                "sample_path": str(sample_path),
                "manifest_entry": sample,
            }

        self._check_shapes(item, sample_path)
        return item

    @staticmethod
    def _check_shapes(item, sample_path):
        pos = item["pos"]
        node_features = item["node_features"]
        target = item["target"]
        triangles = item["triangles"]
        edge_index = item["edge_index"]

        if pos.ndim != 2 or pos.shape[-1] != 2:
            raise ValueError(f"{sample_path}: expected pos shape (N, 2), got {tuple(pos.shape)}")
        if node_features.ndim != 2 or node_features.shape[0] != pos.shape[0]:
            raise ValueError(
                f"{sample_path}: expected node_features shape (N, F), got {tuple(node_features.shape)}"
            )
        if target.shape != (pos.shape[0], 1):
            raise ValueError(f"{sample_path}: expected target shape (N, 1), got {tuple(target.shape)}")
        if triangles.ndim != 2 or triangles.shape[-1] != 3:
            raise ValueError(f"{sample_path}: expected triangles shape (E, 3), got {tuple(triangles.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{sample_path}: expected edge_index shape (2, E), got {tuple(edge_index.shape)}")


def heat2d_batch_size_one_collate(batch):
    """Collate one variable-size Heat2D sample into Transolver's [B, N, C] tensors."""
    if len(batch) != 1:
        raise ValueError(
            "Heat2D currently supports batch_size=1 only. "
            "Padded batching/masked attention is intentionally not implemented yet."
        )

    item = batch[0]
    return {
        "pos": item["pos"].unsqueeze(0),
        "node_features": item["node_features"].unsqueeze(0),
        "target": item["target"].unsqueeze(0),
        "triangles": item["triangles"],
        "element_material_id": item["element_material_id"],
        "nodal_material_id": item["nodal_material_id"],
        "edge_index": item["edge_index"],
        "shape_type": item["shape_type"],
        "shape_type_id": item["shape_type_id"],
        "geometry_metadata_json": item["geometry_metadata_json"],
        "sample_name": item["sample_name"],
        "sample_id": item["sample_id"],
        "sample_path": item["sample_path"],
        "manifest_entry": item["manifest_entry"],
    }
