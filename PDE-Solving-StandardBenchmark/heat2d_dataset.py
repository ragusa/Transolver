import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PARAMETER_NAMES = ("kappa_1", "kappa_2", "q_1", "q_2")
BASE_FEATURE_NAMES = ("material_2_fraction",) + PARAMETER_NAMES


def resolve_heat2d_dataset_roots(dataset_dirs):
    """Resolve one or more Heat2D dataset roots from common launch directories."""
    if isinstance(dataset_dirs, (str, os.PathLike)):
        raw_paths = _split_dataset_roots(str(dataset_dirs))
    else:
        raw_paths = []
        for value in dataset_dirs:
            raw_paths.extend(_split_dataset_roots(str(value)))

    roots = []
    for raw_path in raw_paths:
        path = Path(os.path.expandvars(raw_path)).expanduser()
        if path.exists():
            roots.append(path)
            continue

        benchmark_dir = Path(__file__).resolve().parent
        repo_root = benchmark_dir.parent
        candidates = [
            repo_root / path,
            benchmark_dir / path,
            (benchmark_dir / ".." / path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                roots.append(candidate)
                break
        else:
            raise FileNotFoundError(f"Could not find Heat2D dataset directory: {raw_path}")

    if not roots:
        raise ValueError("At least one Heat2D dataset root is required.")
    return roots


def _split_dataset_roots(value):
    # Accept repeated CLI args, comma-separated paths, or os.pathsep-separated
    # paths. Avoid treating a Windows drive colon as a separator.
    values = []
    for chunk in value.split(","):
        if os.name != "nt" and os.pathsep in chunk:
            values.extend(part for part in chunk.split(os.pathsep) if part)
        elif chunk:
            values.append(chunk)
    return values


class Heat2DDataset(Dataset):
    """Batch-size-one loader for Heat2D FOM samples on variable triangular meshes."""

    def __init__(
        self,
        dataset_dir="fom_generation/data/heat2d_pilot",
        split="train",
        sample_names=None,
        split_seed=0,
        split_fractions=(0.8, 0.1, 0.1),
        max_samples=None,
        include_boundary_mask=False,
        preload=False,
    ):
        self.dataset_roots = resolve_heat2d_dataset_roots(dataset_dir)
        self.split = split
        self.include_boundary_mask = include_boundary_mask
        self.preload = preload
        self._cache = None
        self.input_feature_names = list(BASE_FEATURE_NAMES)
        if include_boundary_mask:
            self.input_feature_names.append("outer_boundary_mask")

        samples = self._discover_samples()
        samples = _assign_sample_level_splits(samples, split_seed, split_fractions)

        if split != "all":
            samples = [sample for sample in samples if sample["split"] == split]
        if sample_names is not None:
            wanted = set(sample_names)
            samples = [sample for sample in samples if sample["sample_name"] in wanted]
        if max_samples is not None:
            samples = samples[: int(max_samples)]

        if not samples:
            roots = ", ".join(str(path) for path in self.dataset_roots)
            raise ValueError(f"No Heat2D samples found for split={split!r} under {roots}")

        self.samples = samples
        if samples[0]["schema"] == "manifest":
            self.input_feature_names = list(samples[0].get("input_feature_names", self.input_feature_names))
        if self.preload:
            self._cache = [self._load_item(index) for index in range(len(self.samples))]

    @property
    def num_input_features(self):
        return len(self.input_feature_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_item(idx)

    def _load_item(self, idx):
        sample = self.samples[idx]
        sample_path = sample["path"]
        with np.load(sample_path, allow_pickle=False) as data:
            if sample["schema"] == "manifest" and "node_features" in data.files:
                item = self._load_manifest_sample(data, sample)
            else:
                item = self._load_fom_sample(data, sample)

        self._check_shapes(item, sample_path)
        return item

    def _discover_samples(self):
        samples = []
        for root in self.dataset_roots:
            fom_samples = sorted(root.glob("geometries/geom_*/sample_*.npz"))
            if fom_samples:
                samples.extend(
                    {
                        "schema": "fom",
                        "root": root,
                        "path": path,
                        "sample_name": _sample_name(root, path),
                    }
                    for path in fom_samples
                )
                continue

            manifest_path = root / "manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                input_feature_names = manifest.get("arrays", {}).get("input_feature_names", [])
                for index, entry in enumerate(manifest.get("samples", [])):
                    sample_file = entry.get("file") or entry.get("path")
                    if sample_file is None:
                        continue
                    path = root / sample_file
                    samples.append(
                        {
                            "schema": "manifest",
                            "root": root,
                            "path": path,
                            "sample_name": entry.get("sample_name", _sample_name(root, path)),
                            "manifest_entry": entry,
                            "manifest_index": index,
                            "input_feature_names": input_feature_names,
                        }
                    )

        if not samples:
            roots = ", ".join(str(path) for path in self.dataset_roots)
            raise FileNotFoundError(
                f"No Heat2D samples found under {roots}. "
                "Expected geometries/geom_*/sample_*.npz or a manifest.json."
            )
        return samples

    def _load_fom_sample(self, data, sample):
        coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        material_id = np.asarray(data["material_id"], dtype=np.int64)
        target = np.asarray(data["T"], dtype=np.float32).reshape(-1, 1)
        params = _sample_parameters(data)
        material_2_fraction = _nodal_material_2_fraction(coordinates.shape[0], triangles, material_id)
        features = [
            material_2_fraction,
            np.full((coordinates.shape[0], 1), params["kappa_1"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["kappa_2"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["q_1"], dtype=np.float32),
            np.full((coordinates.shape[0], 1), params["q_2"], dtype=np.float32),
        ]
        if self.include_boundary_mask:
            features.append(_outer_boundary_mask(coordinates))
        node_features = np.concatenate(features, axis=1)

        geometry_metadata_json = _string_scalar(data["geometry_metadata_json"])
        geometry_metadata = _load_json_scalar(geometry_metadata_json)
        shape_type = geometry_metadata.get("shape", geometry_metadata.get("shape_type", "unknown"))
        sample_id = _int_scalar(data["sample_id"]) if "sample_id" in data.files else sample["path"].stem
        geometry_id = _int_scalar(data["geometry_id"]) if "geometry_id" in data.files else None
        sample_name = f"geom_{geometry_id:05d}_sample_{int(sample_id):05d}" if geometry_id is not None else sample["sample_name"]
        edge_index = _triangle_edge_index(triangles)

        manifest_entry = {
            "split": sample["split"],
            "schema": "fom",
            "root": str(sample["root"]),
            "file": str(sample["path"].relative_to(sample["root"])),
            "geometry_id": geometry_id,
            "sample_id": sample_id,
            "shape_type": shape_type,
            **params,
        }
        return {
            "pos": torch.from_numpy(coordinates),
            "node_features": torch.from_numpy(node_features),
            "target": torch.from_numpy(target),
            "triangles": torch.from_numpy(triangles),
            "element_material_id": torch.from_numpy(material_id),
            "nodal_material_id": torch.from_numpy((material_2_fraction[:, 0] >= 0.5).astype(np.int64) + 1),
            "edge_index": torch.from_numpy(edge_index),
            "shape_type": str(shape_type),
            "shape_type_id": _shape_type_id(shape_type),
            "geometry_metadata_json": geometry_metadata_json,
            "sample_name": sample_name,
            "sample_id": int(sample_id) if isinstance(sample_id, (int, np.integer)) else sample_id,
            "sample_path": str(sample["path"]),
            "manifest_entry": manifest_entry,
        }

    def _load_manifest_sample(self, data, sample):
        entry = sample.get("manifest_entry", {})
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
            "sample_id": int(data["sample_id"]) if "sample_id" in data.files else int(entry.get("sample_id", 0)),
            "sample_path": str(sample["path"]),
            "manifest_entry": {**entry, "split": sample["split"]},
        }
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


def _assign_sample_level_splits(samples, split_seed, split_fractions):
    samples = sorted(samples, key=lambda sample: str(sample["path"]))
    indices = np.arange(len(samples))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)

    train_fraction, val_fraction, _ = split_fractions
    n_total = len(samples)
    if n_total >= 3:
        n_train = max(1, int(round(train_fraction * n_total)))
        n_val = max(1, int(round(val_fraction * n_total)))
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1
    elif n_total == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0

    split_by_index = {}
    for rank, index in enumerate(indices):
        if rank < n_train:
            split = "train"
        elif rank < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_by_index[int(index)] = split

    assigned = []
    for index, sample in enumerate(samples):
        assigned.append({**sample, "split": split_by_index[index]})
    return assigned


def _sample_name(root, path):
    try:
        return str(path.relative_to(root)).replace("\\", "/").replace("/", "__").replace(".npz", "")
    except ValueError:
        return path.stem


def _sample_parameters(data):
    if "param_names" in data.files and "param_values" in data.files:
        names = [_string_scalar(name) for name in np.asarray(data["param_names"])]
        values = np.asarray(data["param_values"], dtype=np.float32)
        params = {name: float(value) for name, value in zip(names, values)}
    else:
        params = {name: float(np.asarray(data[name]).reshape(())) for name in PARAMETER_NAMES if name in data.files}

    missing = [name for name in PARAMETER_NAMES if name not in params]
    if missing:
        raise KeyError(f"Missing Heat2D parameter values: {missing}")
    return params


def _nodal_material_2_fraction(n_nodes, triangles, material_id):
    is_material_2 = (material_id == 2).astype(np.float32)
    sums = np.zeros(n_nodes, dtype=np.float32)
    counts = np.zeros(n_nodes, dtype=np.float32)
    for local_node in range(3):
        nodes = triangles[:, local_node]
        np.add.at(sums, nodes, is_material_2)
        np.add.at(counts, nodes, 1.0)
    counts = np.maximum(counts, 1.0)
    return (sums / counts).reshape(n_nodes, 1).astype(np.float32)


def _outer_boundary_mask(coordinates, tol=1e-6):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    mask = (np.isclose(x, 0.0, atol=tol) | np.isclose(x, 1.0, atol=tol) |
            np.isclose(y, 0.0, atol=tol) | np.isclose(y, 1.0, atol=tol))
    return mask.astype(np.float32).reshape(-1, 1)


def _triangle_edge_index(triangles):
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


def _load_json_scalar(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _string_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _int_scalar(value):
    return int(np.asarray(value).reshape(()))


def _shape_type_id(shape_type):
    shape_ids = {"disk": 0, "square": 1, "triangle": 2}
    return shape_ids.get(str(shape_type), None)
