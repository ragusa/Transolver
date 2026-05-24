"""Smoke tests for LR scheduler integration in exp_heat2d.py.

Tests 1 is a pure argparse unit test (no data needed).
Tests 2-5 are subprocess smoke runs and require the pilot dataset at
fom_generation/data/heat2d_pilot to be present locally.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "PDE-Solving-StandardBenchmark"
PILOT_DATA = REPO_ROOT / "fom_generation" / "data" / "heat2d_pilot"

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from exp_heat2d import build_parser  # noqa: E402


# ── Test 1: argument parsing (no data needed) ─────────────────────────────────

def test_lr_scheduler_choices_and_default():
    parser = build_parser()

    args = parser.parse_args([])
    assert args.lr_scheduler == "none"
    assert args.min_lr == 1e-5
    assert args.lr_plateau_factor == 0.5
    assert args.lr_plateau_patience == 5
    assert args.max_grad_norm is None

    for choice in ("none", "onecycle", "cosine", "reduce_on_plateau"):
        a = parser.parse_args([f"--lr-scheduler={choice}"])
        assert a.lr_scheduler == choice

    with pytest.raises(SystemExit):
        parser.parse_args(["--lr-scheduler=invalid"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _needs_pilot_data():
    return pytest.mark.skipif(
        not PILOT_DATA.exists(),
        reason=f"pilot dataset not found at {PILOT_DATA}",
    )


def _tiny_argv(tmp_path, extra):
    return [
        sys.executable,
        str(BENCHMARK_DIR / "exp_heat2d.py"),
        "--mode", "train",
        "--data_path", str(PILOT_DATA),
        "--output-dir", str(tmp_path),
        "--epochs", "3",
        "--max-train-samples", "4",
        "--max-val-samples", "2",
        "--max-test-samples", "2",
        "--eval-every", "1",
    ] + extra


# ── Test 2: --lr-scheduler none (default behavior unchanged) ──────────────────

@_needs_pilot_data()
def test_scheduler_none_epoch_metrics(tmp_path):
    result = subprocess.run(
        _tiny_argv(tmp_path, ["--lr-scheduler", "none"]),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    csv_path = tmp_path / "epoch_metrics.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3
    assert "current_lr" in rows[0]

    import torch
    ckpt = torch.load(tmp_path / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" not in ckpt


# ── Test 3: --lr-scheduler onecycle — LR varies, saved in checkpoint ──────────

@_needs_pilot_data()
def test_scheduler_onecycle_lr_changes(tmp_path):
    result = subprocess.run(
        _tiny_argv(tmp_path, ["--lr-scheduler", "onecycle", "--lr", "1e-3"]),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    rows = list(csv.DictReader((tmp_path / "epoch_metrics.csv").open()))
    lrs = [float(r["current_lr"]) for r in rows]
    # OneCycleLR steps each optimizer.step(); across 3 epochs with 4 samples
    # the LR should differ from epoch to epoch
    assert not all(lr == lrs[0] for lr in lrs), f"LR did not change: {lrs}"

    import torch
    ckpt = torch.load(tmp_path / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in ckpt


# ── Test 4: --lr-scheduler cosine — LR non-increasing ─────────────────────────

@_needs_pilot_data()
def test_scheduler_cosine_lr_decreases(tmp_path):
    result = subprocess.run(
        _tiny_argv(tmp_path, ["--lr-scheduler", "cosine", "--lr", "1e-3", "--min-lr", "1e-6"]),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    rows = list(csv.DictReader((tmp_path / "epoch_metrics.csv").open()))
    lrs = [float(r["current_lr"]) for r in rows]
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-12, f"LR increased at epoch {i + 1}: {lrs[i - 1]:.4e} -> {lrs[i]:.4e}"


# ── Test 5: --lr-scheduler reduce_on_plateau — completes, lr_scheduler in summary

@_needs_pilot_data()
def test_scheduler_reduce_on_plateau_completes(tmp_path):
    result = subprocess.run(
        _tiny_argv(
            tmp_path,
            ["--lr-scheduler", "reduce_on_plateau", "--lr-plateau-patience", "1", "--lr-plateau-factor", "0.5"],
        ),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["lr_scheduler"] == "reduce_on_plateau"

    # Verify lr_history is in learning_curves.json
    lc = json.loads((tmp_path / "learning_curves.json").read_text())
    assert "lr_history" in lc
    assert len(lc["lr_history"]) == 3
