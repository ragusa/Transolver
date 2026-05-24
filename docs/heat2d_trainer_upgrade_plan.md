# Heat2D Trainer Upgrade Plan

Comparison of `exp_airfoil.py` and `exp_heat2d.py`, and the minimal improvements
needed before conducting architecture sweeps on Heat2D.

---

## 1. Features in exp_airfoil.py to port to Heat2D

| Feature | Where in airfoil | Why port it |
|---------|-----------------|-------------|
| `OneCycleLR` scheduler | line 109; `scheduler.step()` per batch | Can significantly improve convergence and final accuracy |
| Cosine annealing option | — (not present, but standard alternative) | Smoother decay; useful when OneCycleLR cycle length is hard to tune |
| `ReduceLROnPlateau` option | — (not present, but standard alternative) | Adaptive; good for long runs where plateau detection matters |
| Gradient clipping (`--max_grad_norm`) | lines 195–196 | Prevents gradient explosions, especially with variable-size meshes |

---

## 2. Features in exp_airfoil.py NOT to port directly

| Feature | Reason |
|---------|--------|
| `batch_size=8` default | Heat2D uses variable-size unstructured meshes; batch collation requires `batch_size=1` |
| Single checkpoint `{save_name}.pt` | Heat2D already uses `best.pt` / `last.pt` with best-validation selection; superior |
| No validation split | Airfoil conflates test set with validation; Heat2D maintains explicit `train/val/test` splits |
| No per-epoch CSV | Airfoil logs only to stdout; Heat2D's `epoch_metrics.csv` is better for post-hoc analysis |
| `--downsamplex/y` flags | Airfoil-specific structured-grid downsampling; not applicable to unstructured meshes |
| `--unified_pos` / `--ref` tuning | Airfoil-specific; Heat2D already exposes these separately |
| Result PDF visualization | Airfoil writes matplotlib PDFs per sample (eval mode only); Heat2D has richer PNG diagnostics |
| Hard-coded `./checkpoints/` path | Heat2D uses `--output-dir`-relative paths; more flexible |

---

## 3. Features already in exp_heat2d.py that are better than exp_airfoil.py

| Feature | Details |
|---------|---------|
| Explicit train/val/test splits | Three independent loaders; best checkpoint selected on `val` relative L2, not test |
| `best.pt` checkpoint selection | Saves whenever `val_relative_l2` improves; enables fair test evaluation |
| `epoch_metrics.csv` | Per-epoch table with all timing and metric columns; trivially plotted externally |
| `run_summary.json` | Machine-readable summary for sweep aggregation |
| `learning_curves.json` | Full epoch-level time series; deterministic and version-controlled |
| Feature-set metadata | `feature_set`, `input_feature_names`, `fun_dim` stored in config and checkpoint |
| Manifest-style dataset support | `--data_path` accepts multiple directories; supports heterogeneous shape libraries |
| Gradient accumulation | `--grad-accum-steps` allows effective batch sizes > 1 without collating variable meshes |
| Normalization pipeline | Compute-from-train stats; saved to `normalization_stats.json`; restored on eval/resume |
| Baseline comparisons in eval | Zero and train-mean baselines in `run_eval()` provide context for model metrics |
| `--mode smoke` | Minimal end-to-end sanity run for CI |
| Per-sample CSV metrics | Area-weighted, per-shape, and nodal error breakdowns |

---

## 4. Minimal trainer improvements before architecture sweeps

These are the lowest-risk, highest-reward changes before running parameter sweeps:

1. **LR scheduler support** *(this PR)*
   Add `--lr-scheduler {none,onecycle,cosine,reduce_on_plateau}` with `--lr-scheduler none`
   as the default so all existing runs are unchanged. Without a scheduler, AdamW with a
   fixed LR often under-performs; adding OneCycleLR typically yields meaningful accuracy
   improvements at no extra compute cost.

2. **Gradient clipping** *(this PR)*
   Add `--max-grad-norm` (default `None`). Prevents rare but expensive divergence events
   during sweeps with aggressive learning rates or unusual mesh geometries.

3. **Resume support** *(future)*
   Add `--resume` to restart from `last.pt`, restoring optimizer and scheduler state. Not
   strictly required for fresh sweeps but important for long HPC runs that may be
   preempted.

4. **Distributed / multi-GPU** *(future, optional)*
   Not needed for current mesh sizes; note here for when scaling up.
