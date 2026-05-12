"""Print compact summaries from saved Heat2D run artifacts."""

import argparse
import csv
import json
from pathlib import Path


def load_summary(path):
    path = Path(path)
    if path.is_file():
        return _load_file(path)

    run_summary = path / "run_summary.json"
    if run_summary.exists():
        return _load_file(run_summary)

    diagnostics = path / "diagnostics.json"
    if diagnostics.exists():
        data = _load_file(diagnostics)
        overall = data["model"]["overall"]
        summary = {
            "run_type": "eval",
            "path": str(path),
            "checkpoint": data.get("checkpoint"),
            "checkpoint_epoch": data.get("checkpoint_epoch"),
            "overall_relative_l2": overall.get("mean_sample_relative_l2"),
            "area_weighted_relative_l2": overall.get("area_weighted_relative_l2"),
            "zero_prediction_relative_l2": data["baselines"]["zero_prediction"].get("mean_sample_relative_l2"),
            "train_mean_prediction_relative_l2": data["baselines"]["train_mean_prediction"].get(
                "mean_sample_relative_l2"
            ),
            "best_sample": data["model"].get("best_sample"),
            "worst_sample": data["model"].get("worst_sample"),
        }
        for shape, metrics in data["model"].get("per_shape", {}).items():
            summary[f"{shape}_relative_l2"] = metrics.get("relative_l2")
        return summary

    metrics = path / "metrics.json"
    if metrics.exists():
        data = _load_file(metrics)
        return {
            "run_type": "train",
            "path": str(path),
            "train_samples": data.get("train_samples"),
            "val_samples": data.get("val_samples"),
            "test_samples": data.get("test_samples"),
            "final_train_loss": data.get("final_train_loss"),
            "best_validation_relative_l2": data.get("best_validation_relative_l2"),
            "best_checkpoint_epoch": data.get("best_checkpoint_epoch"),
            "test_relative_l2_from_best_validation": data.get("test_relative_l2_from_best_validation"),
            "test_mse_from_best_validation": data.get("test_mse_from_best_validation"),
            "best_checkpoint": data.get("checkpoints", {}).get("best"),
        }

    sweep = path / "sweep_results.csv"
    if sweep.exists():
        rows = _load_csv(sweep)
        best = min(rows, key=lambda row: float(row["test_relative_l2_from_best_validation"]))
        return {
            "run_type": "sweep",
            "path": str(path),
            "num_configs": len(rows),
            "best_config": best,
        }

    dataset_summary = path / "dataset_summary.json"
    if dataset_summary.exists():
        data = _load_file(dataset_summary)
        return {
            "run_type": "dataset",
            "path": str(path),
            "num_samples": data.get("num_samples"),
            "shape_counts": data.get("shape_counts"),
            "split_counts": data.get("split_counts"),
            "shape_counts_per_split": data.get("shape_counts_per_split"),
            "node_count": data.get("node_count"),
            "element_count": data.get("element_count"),
        }

    raise FileNotFoundError(f"No Heat2D summary artifacts found under {path}")


def _load_file(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_parser():
    parser = argparse.ArgumentParser(description="Print a compact JSON summary for a completed Heat2D run.")
    parser.add_argument("path", help="Run directory, dataset directory, or summary JSON/CSV file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = load_summary(args.path)
    print(json.dumps(summary, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
