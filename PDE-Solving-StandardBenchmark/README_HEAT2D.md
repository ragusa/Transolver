# Heat2D Transolver Loader and First Training Path

This adapter loads Heat2D FOM datasets with samples discovered from:

```text
geometries/geom_*/sample_*.npz
```

Multiple dataset roots can be passed after `--data_path`. Sample-level splits are deterministic and use 80% train, 10% validation, and 10% test across the combined discovered sample list.

Each sample is loaded as:

```text
x  = coordinates, shape (n_nodes, 2)
y  = T[:, None], shape (n_nodes, 1)
fx = [material_2_fraction, kappa_1, kappa_2, q_1, q_2], shape (n_nodes, 5)
```

Internally, raw-FOM samples now pass through explicit basic embedding builders.
The default builder preserves the five-feature vector above exactly. Add
`--include-boundary-mask` to select the boundary-mask builder, which appends an
outer-square boundary mask as a sixth feature. The meshes can have different
node counts, so this first training path intentionally supports `--batch-size 1`
only.

## Requirements

Install the Transolver benchmark dependencies before running the Heat2D loader
or training entry point:

```bash
cd PDE-Solving-StandardBenchmark
pip install -r requirements.txt
```

The Heat2D Transolver path uses the irregular-mesh model, so `timm` and
`einops` are required. They are listed in `PDE-Solving-StandardBenchmark/requirements.txt`.

## Vision Dataset Roots

```bash
cd PDE-Solving-StandardBenchmark

DATA_ROOTS=(
  /scratch/user/$USER/heat2d_fom_train_0
  /scratch/user/$USER/heat2d_fom_train_1
  /scratch/user/$USER/heat2d_fom_train_2
  /scratch/user/$USER/heat2d_fom_train_3
  /scratch/user/$USER/heat2d_fom_train_4
)
```

## Loader Smoke Test

```bash
python exp_heat2d.py \
  --mode smoke \
  --data_path "${DATA_ROOTS[@]}" \
  --split all \
  --max-samples 5
```

The command prints the number of loaded samples, representative `x`, `fx`, and `y` shapes, the min/max of `y`, and whether all loaded arrays are finite.

## Performance Knobs

The working path still defaults to `--batch-size 1` because meshes have
different node counts. For medium runs on Vision, try overlapping disk I/O and
host-to-device copies:

```bash
python exp_heat2d.py \
  --mode train \
  --data_path "${DATA_ROOTS[@]}" \
  --epochs 20 \
  --max-train-samples 2000 \
  --max-val-samples 400 \
  --max-test-samples 400 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --batch-size 1 \
  --output-dir results/heat2d_fom_medium
```

For medium-size runs where the selected samples fit comfortably in CPU memory,
add `--preload-data` to read and featurize NPZ samples once at dataset
construction time:

```bash
python exp_heat2d.py \
  --mode train \
  --data_path "${DATA_ROOTS[@]}" \
  --epochs 20 \
  --max-train-samples 2000 \
  --max-val-samples 400 \
  --max-test-samples 400 \
  --preload-data \
  --pin-memory \
  --batch-size 1 \
  --output-dir results/heat2d_fom_medium_cached
```

If `--preload-data` is combined with `--num-workers > 0`, each worker process
can hold its own copy of the cached dataset. Use this only when the memory
budget is clearly safe.

Training now prints `train_seconds`, `train_samples_per_second`,
`epoch_seconds`, and `epoch_samples_per_second` each epoch. The same values are
saved in `learning_curves.json`, `metrics.json`, `run_summary.json`, and
`epoch_metrics.csv`.

## Tiny Training Check

```bash
python exp_heat2d.py \
  --mode train \
  --data_path "${DATA_ROOTS[@]}" \
  --epochs 3 \
  --max-train-samples 100 \
  --max-val-samples 20 \
  --max-test-samples 20 \
  --n-hidden 64 \
  --n-layers 3 \
  --n-heads 4 \
  --slice_num 16 \
  --lr 0.001 \
  --batch-size 1 \
  --output-dir results/heat2d_fom_tiny
```

## Batching Notes

Two simple batching paths are plausible next, but are intentionally not enabled
in this first performance pass:

1. Padding: pad `x`, `fx`, and `y` to the largest node count in each batch and
   carry a node mask. This needs masked loss immediately, and may also need
   masked attention or careful output filtering so padded nodes do not affect
   physics attention.
2. Same-size grouping: group by `geometry_id` or exact `n_nodes`, then batch
   samples from the same geometry/node count. This avoids padding and is a good
   fit for Heat2D because each geometry has multiple parameter samples on the
   same mesh.

The current recommendation is to benchmark `--num-workers`, `--pin-memory`,
and `--preload-data` first, then implement same-geometry batching before padded
masked batching.

## Evaluation

```bash
python exp_heat2d.py \
  --mode eval \
  --data_path "${DATA_ROOTS[@]}" \
  --checkpoint results/heat2d_fom_tiny/checkpoints/best.pt \
  --max-test-samples 20 \
  --eval-output-dir results/heat2d_fom_tiny/diagnostics
```
