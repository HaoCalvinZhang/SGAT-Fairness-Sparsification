# SGAT + Fairness (new independent project)

This project is an independent, *source-guided PyTorch port of the core L0-SGAT algorithm* from Ye & Ji's **Sparse Graph Attention Networks**. It adds **squared soft Demographic Parity** to the training loss while keeping the same source-guided SGAT model and training protocol for **both** beta=0 and beta>0.

**Primary A/B comparison:** beta=0 (original L0-SGAT training objective) versus beta>0 (same model plus DP); no GCN model substitution and **no fixed edge retention**. Both methods share the same graph, split, hyperparameters, evaluation rule, RNG seed, optimizer, and checkpoint procedure.

> Fidelity note: this is a modern **PyTorch scatter-operation port**, not a byte-identical execution of the old DGL 0.4/PyTorch 1.4 repository. The core SGAT math and mechanisms are implemented to match `gat.py`/`train.py`; DGL's graph operations and obsolete dependencies are replaced. Numerical identity with historical runs has NOT been established. See `SOURCE_AUDIT.md` for detailed mapping.

## Reference

- Ye, Y.; Ji, S., *Sparse Graph Attention Networks*, IEEE TKDE. [Paper](https://arxiv.org/abs/1912.00552) | [Original author repository](https://github.com/Yangyeeee/SGAT)
- Original reference paths: `gat.py`, `train.py`, `README.md`.
- Our files were written for this experiment; original repository source files are **not** bundled.

## 0. Install

From the new folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

**Python 3.13:** Should work with the installed current PyTorch on your Mac. This version needs **no DGL and no torch_geometric**. It uses only torch + numpy + pandas + matplotlib; scipy is included for the compatible SBM generator.

## 1. Generate new graph (or copy the exact old `original/` folder)

```bash
python generate.py \
  --n 10000 --degree 20 --hs 0.8 --hy 0.8 \
  --label-signal 0.5 --sensitive-signal 0.5 \
  --feature-noise 1 --graph-seed 42
```

The output graph ID (same format as previous project):

`n10000_d20_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1`

**For strongest controlled comparison across your previous and new experiments**, copy the old `original/graph.npz` and metadata, rather than regenerating and assuming equality. The included generator itself is unchanged from the previous GNN project.

## 2. Smoke test (tiny graph, 4 brief runs)

```bash
python generate.py --n 120 --degree 6 --hs 0.8 --hy 0.8 --label-signal 0.5 --graph-seed 42
python train_sgat_fair.py \
  --graph results/datasets/n120_d6_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 0 1e-6 --betas 0 10 \
  --runs 1 --epochs 5 --patience 3 \
  --hidden 8 --idrop 0 --adrop 0 --device cpu --log-every 2
```

This verifies loading, forward/backprop, saving CSV + selected edges and 11 PNGs. Values from 5 epochs on 120 nodes are *not* research results.

## 3. Main SGAT baseline vs SGAT + fairness (first calibration)

One Lambda and two Betas:

```bash
python train_sgat_fair.py \
  --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 1e-6 \
  --betas 0 10 \
  --runs 1 --log-every 20 --device cpu
```

Then a preliminary **lambda x beta** scan:

```bash
python train_sgat_fair.py \
  --graph results/datasets/n10000_d20_hs0.80_hy0.80_seed42_ay0.5_as0.5_noise1 \
  --lambdas 0 1e-7 3e-7 1e-6 3e-6 1e-5 \
  --betas 0 1 10 100 \
  --runs 1 --log-every 20 --device cpu
```

These are *starting values*, not a guarantee of uniform 10–90% edge keeps. The original SGAT coefficient multiplies **SUM** of gate probabilities including self-loop logits, not mean. Larger lambda encourages deletion; actual edge retention is **learned**. Lambda values are not percentages.

For 5 paired seeds, **run the exact same command**, changing `--runs 1` to `--runs 5`. Previously completed (λ,β,seed) results are skipped within that same model-config folder. Full sweeps on 10k nodes are substantially more expensive than the smoke test.

## 4. Important: definitions and output

Training loss:

\[
\mathcal L=\operatorname{CE}_{V_{train}} + \lambda \sum_{e\in E_{dir}\cup E_{loop}} P(Z_e>0) + \beta\left(\overline{p}_{S=0,V_{train}}-\overline{p}_{S=1,V_{train}}\right)^2.
\]

- `alpha_e` is **not** a free per-edge parameter; logAlpha is a **feature-based, first-layer GAT attention logit** using a shared trainable projection/attention vector and scalar gate bias.
- SGAT trains with **stochastic Concrete gates**; inference uses **temperature-scaled deterministic gates**; self-loops are always active in message passing. Learned attention from layer 1 is reused in all subsequent GAT layers.
- Reference SGAT L0 loss sums probability over directed edges **plus loops**, matching the source first-layer loss. Reported `expected_keep_pct` and `actual_keep_pct` **exclude loops**.
- For a graph with unique undirected `u<v` edges, we create **both directed messages**. `actual_keep_pct` is the proportion of surviving **directed non-self-loop messages**, not the proportion of undirected edges. We also report `undirected_union_keep_pct` and `undirected_both_keep_pct` as diagnostics. Do **not** directly compare this directed ratio with fixed-budget GCN's undirected Edge Keep.
- **Primary test Accuracy/DP/EO** are from the **trained SGAT with deterministic continuous gates**, following source inference. **Secondary `hard_*` metrics** replace positive deterministic gates by exactly 1, representing the actual binarized graph; these can differ. Both use *the same trained SGAT weights*, not a retrained independent GCN.
- `test_dp_gap` uses hard labels, whereas training soft DP uses probabilities. `test_eo_gap` uses label-positive conditional TPR gap; also output FPR/equalized odds.
- Reference-style checkpoint: best **validation Accuracy** (ties update); test split never selects checkpoints or beta. `--checkpoint val_loss` is an optional departure for diagnosis.
- Model architecture starting defaults resemble SGAT GitHub's *Cora* example: 1 hidden layer, 2 heads, hidden 64, input dropout .4, attention dropout .3, lr .01, 200 epochs, checkpoint patience 150; these are **not necessarily optimal for SBM**.

### Terminal output and saved files

Each run prints:

`λ`, `β`, `seed`, **Actual Keep**, **Expected Keep**, **Test Accuracy**, **Test DP Gap**, **EO Gap**, **Hard Accuracy/DP**, epoch and seconds.

Outputs are isolated under:

```text
outputs/<GRAPH_ID>/config_<HASH>/
  config.json
  runs.csv                  # each seed, for every lambda × beta
  summary.csv               # group mean and sample SD
  plots/                    # 11 charts (including fairness/accuracy/edge keep)
  selected_edges/           # directed selected edges; union/both undirected masks
  checkpoints/              # only with --save-checkpoints
```

By default each completed run is written to CSV **immediately** (safe to resume), duplicate settings get skipped, and `--force` overwrites matching results in the CSV after rerunning them. Model architecture and training settings form a separate `<HASH>`, so different checkpoint/epoch/learning-rate configurations won't silently mix.

Optional `--save-gates` outputs all directed-edge alpha values, probabilities and deterministic gates for structural/threshold debugging. It increases output size.

Recreate charts without training, using the printed exact runs.csv path:

```bash
python plot_sgat_fair.py --runs-csv outputs/<GRAPH_ID>/config_<HASH>/runs.csv
```

### Important research caveats

- This clean PyTorch port is intended to reproduce original SGAT **core algorithmic mechanisms**, not exact numerical results on Cora/PPI because we are testing **new SBM graphs** and using a new framework. For publication-grade SGAT replication, additionally benchmark against the authors' original DGL implementation on one of their published datasets.
- Graph-generation metadata includes `label_signal` and `sensitive_signal`; when `hy=.8`, n=10k graph may already be almost perfectly classified. Low original DP can limit measurable improvement; explore `hy=.4/.6`, `ay=.25`, etc. after checking main experiment.
- Both beta0 and beta>0 should be run under the **same λ, seed and architecture**. If beta changes learned edge count, their raw test results don't isolate edge quality; also compare matched/near-matched actual keeps.
- Edge Keep is a proxy for graph size, **not an actual speedup benchmark**. If efficiency is a claim, separately measure training/inference time and peak memory.
