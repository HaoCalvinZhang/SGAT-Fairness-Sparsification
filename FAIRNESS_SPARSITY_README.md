# Fairness + Sparsity ONLY (No Classification Objective)

Run from project root after generating the graph:

```bash
python train_fairness_sparsity.py \
  --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 2e-6 5e-6 1e-5 \
  --betas 1 10 50 100 \
  --runs 1 --epochs 200 --device cpu
```

Loss = lambda * summed L0 + beta * squared soft DP. **No classification loss.**
`--checkpoint last` avoids choosing epochs by validation accuracy.
Results are in `outputs_fairness_sparsity_only/` and do not overwrite existing SGAT results.

**Warning:** Without classification supervision, the network can predict a constant class or probability for everyone and obtain DP=0 without meaningful fairness. `test_pred_positive_rate` and `test_soft_dp_gap` are included to detect this. EO Gap is only diagnostic. The resulting scores should not be treated as a useful node classifier or proof of fairness-aware edge selection.

The original `train_sgat_fair.py` remains unchanged.
