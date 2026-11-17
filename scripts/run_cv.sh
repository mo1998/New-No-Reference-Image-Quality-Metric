#!/usr/bin/env bash
# Headline 5-fold content CV. Trains on all non-test contents.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/siglip_b16_aux_kadid.yaml}
NAME=${2:-siglip_aux_cv65}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src \
  .venv/bin/python -m nriqa.cv --config "$CONFIG" --k 5 --name "$NAME" \
  --device cuda:0 --epochs 18 --output-root runs_cv65 --results-dir results/cv \
  --set data.val_frac=0.001 --set train.select_best=false
