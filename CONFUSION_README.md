# Sensitive-group confusion matrices

This variant preserves the fairness+sparsity-only loss (no CE). After each training run it saves:
- `outputs_fairness_sparsity_only/<graph>/config_<id>/confusion/lambda_..._beta_..._seed_....csv`
- `.../group_confusion_all.csv` (all completed configurations)
- optional `.../predictions/*.npz` with `--save-predictions`.

The CSV has one row for each `prediction_mode` (`deterministic` / `binary`), `split` (`test` / `full`), and `sensitive_group` (`0` / `1`). Includes TP, FP, TN, FN, TPR, FPR, FNR, TNR, predicted positive rate and accuracy.

**Important:** Previous runs with only selected edges cannot be reconstructed; these confusion matrices require a new model training run. Existing configurations are skipped by default. To preserve existing outputs, use a fresh `--output-base outputs_confusion_80_20` rather than `--force`.

Example:
```
python train_fairness_sparsity.py \
 --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.25_as0.5_noise1_s080 \
 --lambdas 5e-6 --betas 50 --runs 1 --epochs 200 \
 --checkpoint last --device cpu --save-predictions \
 --output-base outputs_confusion_80_20
```
