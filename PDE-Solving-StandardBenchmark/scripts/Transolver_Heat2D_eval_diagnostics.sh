#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

"${PYTHON:-python}" exp_heat2d.py \
  --mode eval \
  --data_path ../fom_generation/data/heat2d_moderate_balanced \
  --test-split test \
  --checkpoint results/heat2d_train_moderate/checkpoints/best.pt \
  --eval-output-dir results/heat2d_train_moderate/diagnostics \
  --plot-samples 6
