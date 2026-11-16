#!/usr/bin/env bash
set -euo pipefail
python generate.py \
  --n 10000 --degree 20 --hs 0.8 --hy 0.8 \
  --label-signal 0.5 --sensitive-signal 0.5 \
  --feature-noise 1 --graph-seed 42
