"""Integrity checks and summary reporting for Heat2D NPZ datasets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SOLUTION_FIELD_CANDIDATES = ("T", "target", "solution", "temperature", "u")
COORDINATE_FIELD_CANDIDATES = ("coordinates", "pos")
MATERIAL_FIELD_CANDIDATES = ("material_id", "element_material_id")
PARAMETER_NAMES = ("kappa_1", "kappa_2", "q_1", "q_2")


def check_dataset(dataset_dir, summary_path=None, write_summary=True):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    sample_paths = _sample_paths_from_manifest(dataset_dir, manifest) if manifest else _discover_sample_paths(dataset_dir)

    if not sample_paths:
        raise FileNotFoundError(
            f"No Heat2D sample files found in {dataset_dir}. "
            "Expected manifest.json with samples or geometries/geom_*/sample_*.npz."
        )

    errors = []
    shape_counts = Counter()
    split_counts = Counter()
    split_shape_counts = defaultdict(Counter)
    node_counts = []
    element_counts = []
    parameter_values = defaultdict(list)
    material_tags = set()

    temperature_min = None
    temperature_max = None
    temperature_sum = 0.0
    temperature_count = 0
    all_temperature_finite = True
    all_coordinates_finite = True
    all_samples_have_material_tags_1_and_2 = True
    representative = None

    manifest_samples = manifest.get("samples", []) if manifest else []
    for index, sample_path in enumerate(sample_paths):
        sample_meta = manifest_samples[index] if index < len(manifest_samples) else {}
        try:
            with np.load(sample_path, allow_pickle=False) as data:
                sample_info = _inspect_sample(data, sample_path)
                if representative is None:
                    representative = {
                        "path": str(sample_path),
                        "keys": sample_info["keys"],
                    }

                temperature = sample_info["solution"]
                coordinates = sample_info["coordinates"]
                material_id = sample_info["material_id"]
                sample_material_tags = {int(tag) for tag in np.unique(material_id)}

                all_temperature_finite = all_temperature_finite and bool(np.all(np.isfinite(temperature)))
                all_coordinates_finite = all_coordinates_finite and bool(np.all(np.isfinite(coordinates)))
                all_samples_have_material_tags_1_and_2 = (
                    all_samples_have_material_tags_1_and_2 and {1, 2}.issubset(sample_material_tags)
                )
                if temperature.size:
                    temperature_min = _none_min(temperature_min, float(np.min(temperature)))
                    temperature_max = _none_max(temperature_max, float(np.max(temperature)))
                    temperature_sum += float(np.sum(temperature, dtype=np.float64))
                    temperature_count += int(temperature.size)

                material_tags.update(sample_material_tags)
                node_counts.append(int(coordinates.shape[0]))
                if "triangles" in data.files:
                    element_counts.append(int(data["triangles"].shape[0]))

                shape = _sample_shape(data, sample_meta)
                if shape is not None:
                    shape_counts[shape] += 1
                    split = str(sample_meta.get("split", "unknown"))
                    split_counts[split] += 1
                    split_shape_counts[split][shape] += 1

                for name, value in _sample_parameters(data).items():
                    parameter_values[name].append(float(value))
        except Exception as exc:  # noqa: BLE001 - report all integrity failures together.
            errors.append(f"{sample_path}: {exc}")

    if errors:
        joined = "\n".join(errors)
        raise AssertionError(f"Heat2D dataset integrity check failed:\n{joined}")

    geometry_dirs = sorted((dataset_dir / "geometries").glob("geom_*"))
    temperature_stats = {
        "min": temperature_min,
        "mean": None if temperature_count == 0 else temperature_sum / temperature_count,
        "max": temperature_max,
    }
    parameter_ranges = {
        name: _min_mean_max(parameter_values.get(name, []))
        for name in sorted(parameter_values)
    }

    summary = {
        "dataset_root": str(dataset_dir),
        "output_directory": str(dataset_dir),
        "manifest_path": str(manifest_path) if manifest else None,
        "source": "manifest" if manifest else "discovered",
        "num_geometry_dirs": len([path for path in geometry_dirs if path.is_dir()]),
        "num_sample_files": len(sample_paths),
        "num_samples": len(sample_paths),
        "representative_sample": representative,
        "solution_field": representative_solution_field(representative),
        "temperature_stats": temperature_stats,
        "all_T_finite": all_temperature_finite,
        "all_coordinates_finite": all_coordinates_finite,
        "material_tags": sorted(material_tags),
        "material_tags_1_and_2_present": {1, 2}.issubset(material_tags),
        "all_samples_have_material_tags_1_and_2": all_samples_have_material_tags_1_and_2,
        "parameter_ranges": parameter_ranges,
        "shape_counts": dict(sorted(shape_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "shape_counts_per_split": {
            split: dict(sorted(counts.items())) for split, counts in sorted(split_shape_counts.items())
        },
        "node_count": _min_mean_max(node_counts),
        "element_count": _min_mean_max(element_counts),
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


def _load_manifest(manifest_path):
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sample_paths_from_manifest(dataset_dir, manifest):
    paths = []
    for sample in manifest.get("samples", []):
        sample_file = sample.get("file") or sample.get("path")
        if sample_file is None:
            continue
        paths.append(dataset_dir / sample_file)
    return paths


def _discover_sample_paths(dataset_dir):
    return sorted(dataset_dir.glob("geometries/geom_*/sample_*.npz"))


def _inspect_sample(data, sample_path):
    solution_field = _first_present(data, SOLUTION_FIELD_CANDIDATES)
    coordinate_field = _first_present(data, COORDINATE_FIELD_CANDIDATES)
    material_field = _first_present(data, MATERIAL_FIELD_CANDIDATES)
    if solution_field is None:
        raise AssertionError(f"missing solution field; expected one of {SOLUTION_FIELD_CANDIDATES}")
    if coordinate_field is None:
        raise AssertionError(f"missing coordinate field; expected one of {COORDINATE_FIELD_CANDIDATES}")
    if material_field is None:
        raise AssertionError(f"missing material field; expected one of {MATERIAL_FIELD_CANDIDATES}")

    coordinates = np.asarray(data[coordinate_field])
    solution = np.asarray(data[solution_field])
    material_id = np.asarray(data[material_field])
    triangles = np.asarray(data["triangles"]) if "triangles" in data.files else None

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise AssertionError(f"{coordinate_field} must have shape (N, 2), got {coordinates.shape}")
    if solution.shape[0] != coordinates.shape[0]:
        raise AssertionError(
            f"{solution_field} first dimension must match coordinates; "
            f"got {solution.shape} and {coordinates.shape}"
        )
    if triangles is not None:
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise AssertionError(f"triangles must have shape (E, 3), got {triangles.shape}")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= coordinates.shape[0]):
            raise AssertionError("triangles contain node indices outside coordinates")
        if material_id.shape != (triangles.shape[0],):
            raise AssertionError(f"{material_field} must align with triangles")

    return {
        "solution": solution,
        "coordinates": coordinates,
        "material_id": material_id,
        "keys": _keys_and_shapes(data),
    }


def _first_present(data, names):
    for name in names:
        if name in data.files:
            return name
    return None


def _keys_and_shapes(data):
    keys = {}
    for key in data.files:
        shape = data[key].shape
        keys[key] = "scalar" if shape == () else list(shape)
    return keys


def representative_solution_field(representative):
    if representative is None:
        return None
    keys = representative["keys"]
    for name in SOLUTION_FIELD_CANDIDATES:
        if name in keys:
            return name
    return None


def _sample_shape(data, sample_meta):
    if "shape_type" in data.files:
        return str(_scalar(data["shape_type"]))
    if "shape" in sample_meta:
        return str(sample_meta["shape"])
    return None


def _sample_parameters(data):
    if "param_names" in data.files and "param_values" in data.files:
        names = [_string_value(name) for name in np.asarray(data["param_names"])]
        values = np.asarray(data["param_values"], dtype=float)
        return {name: float(value) for name, value in zip(names, values)}

    params = {}
    for name in PARAMETER_NAMES:
        if name in data.files:
            params[name] = float(_scalar(data[name]))
    return params


def _scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    raise ValueError(f"expected scalar, got shape {array.shape}")


def _string_value(value):
    value = _scalar(value) if np.asarray(value).shape == () else value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _none_min(current, value):
    return value if current is None else min(current, value)


def _none_max(current, value):
    return value if current is None else max(current, value)


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
