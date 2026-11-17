#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src .venv/bin/python -m nriqa.benchmark --metrics brisque niqe musiq clipiqa arniqa liqe \
  --datasets kadid_test tid2013 --device cuda:0 --batch-size 8
