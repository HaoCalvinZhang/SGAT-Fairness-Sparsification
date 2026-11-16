# SGAT Fairness-Aware Graph Sparsification

A research-oriented PyTorch implementation of **Sparse Graph Attention Networks (SGAT)** with configurable synthetic graph generation, sparsity regularization, demographic-parity regularization, and sensitive-group evaluation. The repository contains **two distinct training objectives**; they must not be confused.

> **Research status:** Experimental code for controlled ablation studies, not a validated fairness guarantee. This is a modern PyTorch adaptation informed by [Ye & Ji, *Sparse Graph Attention Networks*](https://arxiv.org/abs/1912.00552) and the [authors' repository](https://github.com/Yangyeeee/SGAT), not a byte-identical reproduction. See `SOURCE_AUDIT.md`.

## 1. Setup

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run commands from the repository root. The project uses PyTorch, NumPy, Pandas, Matplotlib and SciPy; it does not require DGL or PyTorch Geometric.

## 2. Synthetic graph generator

`generate.py` builds an undirected four-block stochastic block model (SBM), with blocks defined by sensitive attribute `S ∈ {0,1}` and binary class label `Y ∈ {0,1}`. The graph is stored as unique undirected edges; the SGAT trainer constructs directed message edges from these.

### Generator parameters

| Argument | Default | Meaning |
|---|---:|---|
| `--n` | `10000` | Number of nodes |
| `--degree` | `20` | Target **average undirected degree**; realized degree varies slightly |
| `--hs` | `0.8` | Target sensitive homophily: fraction of edges whose endpoints have the same `S` |
| `--hy` | `0.2` | Target label homophily: fraction of edges whose endpoints have the same `Y` |
| `--sensitive-ratio` | `0.8` | Fraction of nodes assigned to `S=0`; `S=1` receives the remainder |
| `--label-signal` | `0.5` | Magnitude of the class-label signal in the first node feature |
| `--sensitive-signal` | `0.5` | Magnitude of the sensitive-attribute signal in the second node feature |
| `--feature-noise` | `1.0` | Gaussian feature noise standard deviation |
| `--feature-dim` | `2` | Number of node features; **this implementation requires exactly 2** |
| `--graph-seed` | `42` | Seed for graph topology, group/label assignment and feature generation |
| `--out` | `results` | Root directory for generated datasets |
| `--overwrite` | off | Regenerate an existing graph instead of reusing it |

**Important:** The generator's default `--hy` is **0.2**, not 0.8. Specify `--hy 0.8` explicitly to reproduce our common high-label-homophily setting.

### What does homophily mean?

- `Hs = P(S_u = S_v | (u,v) is an edge)`: sensitive homophily.
- `Hy = P(Y_u = Y_v | (u,v) is an edge)`: label homophily.
- High `Hs` means edges tend to connect nodes in the same sensitive group; high `Hy` means edges tend to connect nodes with the same class label.
- `Hs` and `Hy` are separate **generation targets**, not trainable model weights. The generator prints the **realized** values after sampling; these can differ slightly from the requested targets.

The generator assigns labels approximately 50/50 **within each sensitive group** (exact for even group sizes). Thus changing `--sensitive-ratio` does **not** automatically introduce a label-prevalence difference between sensitive groups.

**Feasibility constraint for this generator:** `hs >= abs(2*sensitive_ratio - 1)`. For example, with `--sensitive-ratio 0.8`, `--hs 0.8` is feasible, but `--hs 0.4` is rejected. To study `Hs = 0.2, 0.4, 0.6, 0.8` in one sweep, use `--sensitive-ratio 0.5` (or redesign the graph generator). The requested degree and block probabilities must also be feasible.

### Node feature model

With independent standard-normal noise `ε0, ε1`:

```text
x[:, 0] = label_signal     * (2*Y - 1) + feature_noise * ε0
x[:, 1] = sensitive_signal * (2*S - 1) + feature_noise * ε1
```

The generator does not z-score these features. Increasing `label-signal` makes the class label easier to infer from features; increasing `sensitive-signal` makes group membership easier to infer. These parameters are **not** homophily parameters.

### Data split

Train / validation / test = **60% / 20% / 20%**, stratified by `(S,Y)` blocks. With `n=10000` and `--sensitive-ratio 0.8`, the full graph contains 8,000 nodes with `S=0` and 2,000 with `S=1`; the test set contains 1,600 and 400 respectively.

### Example A — 80/20 groups, high homophily for both attributes

```bash
python generate.py \
  --n 10000 --degree 20 \
  --hs 0.8 --hy 0.8 \
  --sensitive-ratio 0.8 \
  --label-signal 0.25 --sensitive-signal 0.5 \
  --feature-noise 1 --graph-seed 42
```

Expected graph directory:

```text
results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.25_as0.5_noise1_s080/
  original/graph.npz
  original/metadata.json
```

`metadata.json` records requested settings and realized graph statistics.

### Example B — 50/50 groups with different homophily settings

```bash
python generate.py --n 10000 --degree 20 --sensitive-ratio 0.5 \
  --hs 0.2 --hy 0.8 --label-signal 0.25 --graph-seed 42
```

To compare homophily conditions, regenerate a graph for each combination, then pass the corresponding generated directory to the trainer. Different graph IDs keep experiments separate.

## 3. Training objectives

### A. Fairness + sparsity only (ablation)

**Script:** `train_fairness_sparsity.py`

\[
L = \lambda \sum_e P(Z_e > 0) + \beta\left(\overline{p}_{S=0,\mathrm{train}} - \overline{p}_{S=1,\mathrm{train}}\right)^2
\]

- `λ` (`--lambdas`) controls L0 edge sparsity. This is a **sum** over gate nonzero probabilities, not an average; values such as `2e-6` are weights, **not retention percentages**.
- `β` (`--betas`) controls squared *soft* demographic-parity regularization, computed from predicted positive-class probabilities on training nodes.
- **No cross-entropy classification loss is used in this script.** Accuracy is computed for evaluation only.
- **Do not interpret a low DP gap alone as successful fair classification.** The model may predict almost every node as class 0 or class 1 (prediction collapse).
- The pair `λ=0, β=0` is invalid for this script because the training objective would be zero.

**25-configuration sweep (5 λ × 5 β; one seed):**

```bash
python train_fairness_sparsity.py \
  --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.25_as0.5_noise1_s080 \
  --lambdas 2e-6 3e-6 5e-6 1e-5 2e-5 \
  --betas 1 10 50 100 1000 \
  --runs 1 --epochs 200 --checkpoint last \
  --device cpu --save-predictions \
  --output-base outputs_confusion_80_20
```

**Five seeds:** repeat the command with `--runs 5`. Runs use consecutive seeds beginning at `--train-seed` (default `1042`). Previously completed `(λ, β, seed)` combinations in the same configuration directory are skipped unless `--force` is supplied.

### B. Classification + fairness + sparsity

**Script:** `train_sgat_fair.py`

\[
L = L_{CE} + \lambda L_{L0} + \beta L_{DP}
\]

Use this script when classification quality must be optimized during training. Its checkpoint options and output conventions differ from the ablation script: run `python train_sgat_fair.py --help` and consult the existing source documentation before comparing the two objectives. The two scripts are **not** interchangeable experimental baselines.

## 4. Training parameters (fairness+sparsity-only script)

| Argument | Default | Meaning |
|---|---:|---|
| `--graph` | required | Graph directory containing `original/graph.npz` |
| `--lambdas` | several | One or more L0 sparsity weights (Cartesian-product sweep with betas) |
| `--betas` | `0 1 10` | One or more fairness weights; `β=0` means **no fairness loss**, not original classification-trained SGAT |
| `--runs` | `1` | Number of training seeds per λ/β pair |
| `--train-seed` | `1042` | Starting training seed |
| `--epochs` | `200` | Maximum training epochs |
| `--patience` | `150` | Early-stopping patience when using `val_objective` checkpointing |
| `--checkpoint` | `last` | `last` uses final epoch; `val_objective` selects by validation fairness+sparsity objective |
| `--hidden` | `64` | Hidden representation size |
| `--num-heads` | `2` | Attention heads in hidden layers |
| `--num-layers` | `1` | Number of hidden SGAT layers |
| `--out-heads` | `1` | Output attention heads |
| `--idrop` | `0.4` | Input/feature dropout |
| `--adrop` | `0.3` | Attention dropout |
| `--lr` | `0.01` | Adam learning rate |
| `--weight-decay` | `0` | Optimizer weight decay |
| `--bias` | `0` | Gate-logit bias parameter |
| `--temperature` | `0.66` | Hard-Concrete relaxation temperature |
| `--residual` | off | Enable residual connections |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--log-every` | `20` | Training-log interval |
| `--output-base` | `outputs_fairness_sparsity_only` | Output directory root |
| `--save-predictions` | off | Save per-node labels and predictions in `.npz` |
| `--save-gates` | off | Save gate diagnostics |
| `--save-checkpoints` | off | Save trained model checkpoints |
| `--plots-only` | off | Regenerate plots from saved runs without training |
| `--force` | off | Recompute existing combinations |

## 5. Evaluation and saved results

For each λ/β/seed combination, the script evaluates:

| Metric | Interpretation |
|---|---|
| Accuracy (%) | Fraction of correctly classified nodes |
| DP gap | Absolute difference in **predicted positive rates** between S=0 and S=1 |
| EO gap | Absolute difference in **true positive rates** between S=0 and S=1 (equal opportunity) |
| Soft DP gap | Absolute difference in average positive-class probabilities between groups |
| Expected edge keep (%) | Mean gate nonzero probability over non-self-loop directed edges |
| Actual edge keep (%) | Fraction of retained non-self-loop **directed** edges under the implementation's deterministic gate rule |
| Positive prediction rate | Fraction of nodes predicted positive; useful for detecting collapse |
| TP / FP / TN / FN | Confusion counts by sensitive group |
| TPR / FPR / FNR / TNR | Group-specific classification rates |

**Directed vs undirected:** Each original undirected edge becomes two directed message edges. Directed edge retention is **not** identical to undirected union/both-edge retention. SGAT message passing keeps self-loops active; reported edge-keep percentages exclude self-loops.

**Prediction modes:** `deterministic` uses continuous deterministic edge gates; `binary` uses hard 0/1 gates. These can produce different predictions. **Evaluation splits:** `test` reports held-out nodes, while `full` reports all nodes (train+validation+test). GNN message passing may still use the full graph when evaluating test nodes.

**Output structure:**

```text
outputs_confusion_80_20/<GRAPH_ID>/config_<HASH>/
├── config.json
├── runs.csv                  # per λ, β, seed: aggregate test metrics
├── summary.csv               # means and sample standard deviations across seeds
├── group_confusion_all.csv   # S=0 / S=1; test/full; deterministic/binary
├── confusion/                # one group-confusion CSV per completed run
├── predictions/              # optional; requires --save-predictions
├── selected_edges/           # retained edge indices
├── checkpoints/              # optional; requires --save-checkpoints
└── plots/                    # generated figures
```

`group_confusion_all.csv` has separate rows for every combination of `lambda_l0`, `beta`, `run_seed`, `prediction_mode`, `split`, and `sensitive_group`. For evaluating the **final discretized sparse graph on held-out nodes**, filter to `split=test` and `prediction_mode=binary`. To compare with the script's main continuous-gate metrics, use `split=test` and `prediction_mode=deterministic`.

If `--runs 1`, sample standard deviation is undefined (`SD n/a`). Use multiple seeds for reliable comparisons.

## 6. Recommended experimental design

1. Hold `n`, degree, group ratio, feature signals and graph seed fixed; vary `Hs` and `Hy` within the feasible region.
2. On each graph, sweep `λ` and `β` using matched training seeds.
3. Compare **Accuracy + DP gap + EO gap + edge retention** jointly, rather than optimizing one number in isolation.
4. Examine group-specific TPR/FPR and positive prediction rates for prediction collapse.
5. Repeat with several graph seeds when assessing robustness to topology generation.

**Interpretation caveat:** Group size imbalance (e.g., 80/20) is not itself demographic-parity unfairness. Here labels remain balanced within sensitive groups by construction. Also, lower DP or EO gaps can result from a degenerate classifier that predicts one class for everyone.

## 7. Reproducibility and references

- The graph generator records requested settings and realized statistics in `original/metadata.json`.
- `config.json` records training configuration and the selected loss.
- Generated graphs and experiment outputs should generally be excluded from Git using `.gitignore`.
- Source paper: [Ye & Ji, Sparse Graph Attention Networks](https://arxiv.org/abs/1912.00552).
- Reference implementation: [Yangyeeee/SGAT](https://github.com/Yangyeeee/SGAT).

For additional implementation details, see `SOURCE_AUDIT.md`, `CONFUSION_README.md`, and the training scripts' `--help` output.
