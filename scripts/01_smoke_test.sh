#!/usr/bin/env bash
set -euo pipefail
python generate.py --n 120 --degree 6 --hs 0.8 --hy 0.8 --label-signal 0.5 --graph-seed 42
python -u train_sgat_fair.py \
  --graph results/datasets/n120_d6_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 0 1e-6 --betas 0 10 --runs 1 \
  --epochs 5 --patience 3 --hidden 8 --idrop 0 --adrop 0 --device cpu --log-every 2
