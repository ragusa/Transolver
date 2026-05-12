# Heat2D Transolver Runbook

This runbook is for external terminal runs. It keeps Codex context small: run experiments in a shell, then paste only a compact summary back into Codex.

All commands below are from the repository root unless noted. On this Windows environment, use:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONPATH=(Get-Location).Path
```

Use the local Python that has the dependencies installed, for example:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' --version
```

## Generate Moderate Dataset

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.generate_balanced_dataset `
  --out fom_generation/data/heat2d_moderate_balanced `
  --samples-per-shape 30 `
  --seed 20260512 `
  --mesh-size 0.14 `
  --safety-margin 0.06 `
  --plots-per-shape 1
```

Output:

- `fom_generation/data/heat2d_moderate_balanced/manifest.json`
- `fom_generation/data/heat2d_moderate_balanced/dataset_summary.json`
- one authoritative `.npz` file per sample under `samples/`

## Check Dataset

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m fom_generation.heat2d.check_dataset `
  --dataset fom_generation/data/heat2d_moderate_balanced
```

The checker verifies required fields, shapes, finite values, material IDs, valid triangles, `edge_index`, and shape metadata. It writes or updates:

- `dataset_summary.json`

## Train One Model

```powershell
Set-Location PDE-Solving-StandardBenchmark
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\ProgramData\anaconda3\python.exe' exp_heat2d.py `
  --mode train `
  --data_path ../fom_generation/data/heat2d_moderate_balanced `
  --train-split train `
  --val-split val `
  --test-split test `
  --epochs 50 `
  --n-hidden 64 `
  --n-layers 3 `
  --n-heads 4 `
  --slice_num 16 `
  --lr 0.001 `
  --batch-size 1 `
  --grad-accum-steps 1 `
  --plot-samples 6 `
  --output-dir results/heat2d_train_moderate
```

Output directory contents:

- `config.json`
- `normalization_stats.json`
- `metrics.json`
- `metrics.csv`
- `run_summary.json`
- `per_sample_metrics.csv`
- `learning_curves.json`
- `learning_curves.png`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `plots/<sample_name>/{target_temperature,predicted_temperature,absolute_error}.png`

The best checkpoint is selected by validation relative L2. Final test metrics are computed from that best validation checkpoint.

## Run Diagnostics

```powershell
Set-Location PDE-Solving-StandardBenchmark
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\ProgramData\anaconda3\python.exe' exp_heat2d.py `
  --mode eval `
  --data_path ../fom_generation/data/heat2d_moderate_balanced `
  --test-split test `
  --checkpoint results/heat2d_train_moderate/checkpoints/best.pt `
  --eval-output-dir results/heat2d_train_moderate/diagnostics `
  --plot-samples 6
```

Diagnostics output:

- `diagnostics.json`
- `run_summary.json`
- `metrics.csv`
- `per_sample_metrics.csv`
- `plots/<sample_name>/{target_temperature,predicted_temperature,absolute_error}.png`

Diagnostics include nodal metrics, per-shape metrics, optional triangle-area-weighted metrics, zero-prediction baseline, and train-mean baseline.

## Run Small Sweep

```powershell
Set-Location PDE-Solving-StandardBenchmark
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\ProgramData\anaconda3\python.exe' sweep_heat2d.py `
  --data_path ../fom_generation/data/heat2d_moderate_balanced `
  --output-dir results/heat2d_sweep_small `
  --epochs 50 `
  --lr 0.001 `
  --slice_num 16 `
  --plot-samples 3
```

Sweep output:

- `sweep_results.csv`
- `sweep_summary.csv`
- `sweep_results.json`
- `run_summary.json`
- `sweep_comparison.png`
- one full train output directory per configuration

## Print Compact Summary

Use this after any completed run, then paste the output back into Codex.

Training run:

```powershell
Set-Location PDE-Solving-StandardBenchmark
& 'C:\ProgramData\anaconda3\python.exe' summarize_heat2d_run.py results/heat2d_train_moderate --pretty
```

Diagnostics run:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' summarize_heat2d_run.py results/heat2d_train_moderate/diagnostics --pretty
```

Sweep:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' summarize_heat2d_run.py results/heat2d_sweep_small --pretty
```

Dataset:

```powershell
Set-Location ..
& 'C:\ProgramData\anaconda3\python.exe' PDE-Solving-StandardBenchmark/summarize_heat2d_run.py fom_generation/data/heat2d_moderate_balanced --pretty
```

## Shell Wrappers

These wrappers mirror the commands above:

- `PDE-Solving-StandardBenchmark/scripts/Transolver_Heat2D_train_moderate.sh`
- `PDE-Solving-StandardBenchmark/scripts/Transolver_Heat2D_eval_diagnostics.sh`
- `PDE-Solving-StandardBenchmark/scripts/Transolver_Heat2D_sweep_small.sh`

They are most convenient from Git Bash or WSL. PowerShell commands above are the most explicit on Windows.

## Current Constraints

- Dataset schema remains one `.npz` per sample plus `manifest.json`.
- Transolver uses `batch_size=1`.
- No padded batching or masked attention.
- `edge_index` remains diagnostic metadata only.
- `Transolver_Irregular_Mesh.py` is not modified.
