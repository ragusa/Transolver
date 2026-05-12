#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

"${PYTHON:-python}" exp_heat2d.py \
  --mode train \
  --data_path ../fom_generation/data/heat2d_pilot \
  --train-split train \
  --val-split test \
  --test-split test \
  --epochs 20 \
  --n-hidden 64 \
  --n-layers 3 \
  --n-heads 4 \
  --slice_num 16 \
  --lr 0.001 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --plot-samples 2 \
  --output-dir results/heat2d_train_small
