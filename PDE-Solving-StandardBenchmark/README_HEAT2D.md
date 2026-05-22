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

Add `--include-boundary-mask` to append an outer-square boundary mask as a sixth feature. The meshes can have different node counts, so this first training path intentionally supports `--batch-size 1` only.

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

## Evaluation

```bash
python exp_heat2d.py \
  --mode eval \
  --data_path "${DATA_ROOTS[@]}" \
  --checkpoint results/heat2d_fom_tiny/checkpoints/best.pt \
  --max-test-samples 20 \
  --eval-output-dir results/heat2d_fom_tiny/diagnostics
```
