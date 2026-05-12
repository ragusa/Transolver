import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = [
    {"name": "h64_l3_heads4", "n_hidden": 64, "n_layers": 3, "n_heads": 4},
    {"name": "h96_l4_heads4", "n_hidden": 96, "n_layers": 4, "n_heads": 4},
    {"name": "h128_l4_heads4", "n_hidden": 128, "n_layers": 4, "n_heads": 4},
]


def main():
    import argparse

    parser = argparse.ArgumentParser("Small Heat2D Transolver hyperparameter sweep")
    parser.add_argument("--data_path", default="../fom_generation/data/heat2d_moderate_balanced")
    parser.add_argument("--output-dir", default="results/heat2d_sweep_small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--slice_num", type=int, default=16)
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    for config in CONFIGS:
        run_dir = output_dir / config["name"]
        command = [
            sys.executable,
            "exp_heat2d.py",
            "--mode",
            "train",
            "--data_path",
            args.data_path,
            "--train-split",
            "train",
            "--val-split",
            "val",
            "--test-split",
            "test",
            "--epochs",
            str(args.epochs),
            "--n-hidden",
            str(config["n_hidden"]),
            "--n-layers",
            str(config["n_layers"]),
            "--n-heads",
            str(config["n_heads"]),
            "--slice_num",
            str(args.slice_num),
            "--lr",
            str(args.lr),
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "1",
            "--plot-samples",
            str(args.plot_samples),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(run_dir),
        ]
        print("running:", " ".join(command))
        start = time.time()
        subprocess.run(command, check=True, env=env)
        runtime = time.time() - start

        with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "name": config["name"],
                "n_hidden": config["n_hidden"],
                "n_layers": config["n_layers"],
                "n_heads": config["n_heads"],
                "slice_num": args.slice_num,
                "epochs": args.epochs,
                "best_checkpoint_epoch": metrics["best_checkpoint_epoch"],
                "best_validation_relative_l2": metrics["best_validation_relative_l2"],
                "test_relative_l2_from_best_validation": metrics["test_relative_l2_from_best_validation"],
                "test_mse_from_best_validation": metrics["test_mse_from_best_validation"],
                "final_train_loss": metrics["final_train_loss"],
                "runtime_seconds": runtime,
                "checkpoint": metrics["checkpoints"]["best"],
                "plot_dir": str(run_dir / "plots"),
            }
        )

    _write_csv(output_dir / "sweep_results.csv", rows)
    _write_csv(output_dir / "sweep_summary.csv", rows)
    with (output_dir / "sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _plot_comparison(output_dir / "sweep_comparison.png", rows)

    best = min(rows, key=lambda row: row["test_relative_l2_from_best_validation"])
    run_summary = {
        "run_type": "sweep",
        "output_dir": str(output_dir),
        "num_configs": len(rows),
        "epochs": args.epochs,
        "data_path": args.data_path,
        "best_config": best,
        "sweep_results_csv": str(output_dir / "sweep_results.csv"),
        "sweep_summary_csv": str(output_dir / "sweep_summary.csv"),
        "comparison_plot": str(output_dir / "sweep_comparison.png"),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"sweep_results: {output_dir / 'sweep_results.csv'}")
    print(f"sweep_summary: {output_dir / 'sweep_summary.csv'}")
    print(f"comparison_plot: {output_dir / 'sweep_comparison.png'}")
    print(f"run_summary: {output_dir / 'run_summary.json'}")
    print(f"best_config: {best}")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(path, rows):
    labels = [row["name"] for row in rows]
    val = [row["best_validation_relative_l2"] for row in rows]
    test = [row["test_relative_l2_from_best_validation"] for row in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.18 for i in x], val, width=0.36, label="best val relative L2")
    ax.bar([i + 0.18 for i in x], test, width=0.36, label="test relative L2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("relative L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
