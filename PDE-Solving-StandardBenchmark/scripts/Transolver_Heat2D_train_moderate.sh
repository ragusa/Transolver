#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

PY="${PYTHON:-python}"
DATASET_DIR="fom_generation/data/heat2d_moderate_balanced"

if [ ! -f "$DATASET_DIR/manifest.json" ]; then
  "$PY" -m fom_generation.heat2d.cli.generate_balanced_dataset \
    --out "$DATASET_DIR" \
    --samples-per-shape 30 \
    --seed 20260512 \
    --mesh-size 0.14 \
    --safety-margin 0.06 \
    --plots-per-shape 1
fi

"$PY" -m fom_generation.heat2d.cli.check_dataset \
  --dataset "$DATASET_DIR"

cd PDE-Solving-StandardBenchmark

"$PY" exp_heat2d.py \
  --mode train \
  --data_path ../fom_generation/data/heat2d_moderate_balanced \
  --train-split train \
  --val-split val \
  --test-split test \
  --epochs 50 \
  --n-hidden 64 \
  --n-layers 3 \
  --n-heads 4 \
  --slice_num 16 \
  --lr 0.001 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --plot-samples 6 \
  --output-dir results/heat2d_train_moderate
