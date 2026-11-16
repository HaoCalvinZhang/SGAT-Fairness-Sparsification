#!/usr/bin/env bash
set -euo pipefail
python -u train_sgat_fair.py \
  --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 0 1e-7 3e-7 1e-6 3e-6 1e-5 \
  --betas 0 1 10 100 --runs 1 --log-every 20 --device cpu
