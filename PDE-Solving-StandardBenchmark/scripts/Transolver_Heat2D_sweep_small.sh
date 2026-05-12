#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

"${PYTHON:-python}" sweep_heat2d.py \
  --data_path ../fom_generation/data/heat2d_moderate_balanced \
  --output-dir results/heat2d_sweep_small \
  --epochs 50 \
  --lr 0.001 \
  --slice_num 16 \
  --plot-samples 3
