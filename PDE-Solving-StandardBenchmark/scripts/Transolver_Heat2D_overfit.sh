#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

"${PYTHON:-python}" exp_heat2d.py \
  --data_path ../fom_generation/data/heat2d_pilot \
  --split train \
  --steps 300 \
  --n-hidden 64 \
  --n-layers 3 \
  --n-heads 4 \
  --slice_num 16 \
  --lr 0.001 \
  --plot-dir results/heat2d_overfit
