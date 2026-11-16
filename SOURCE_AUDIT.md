# SGAT source-to-port audit

Checked official public repository: https://github.com/Yangyeeee/SGAT

Source links:
- https://raw.githubusercontent.com/Yangyeeee/SGAT/master/gat.py
- https://raw.githubusercontent.com/Yangyeeee/SGAT/master/train.py
- https://raw.githubusercontent.com/Yangyeeee/SGAT/master/README.md

| Core mechanism | Original official code | This port |
| --- | --- | --- |
| Architecture | `GAT` hidden + output attention layers | `SGAT.layers` hidden + output layers |
| Attention logits | First layer projected node features `fc`, `attn_l`, `attn_r` and `bias_l0` | `SparseAttentionLayer.gate_logits_first_head` |
| Shared sparsity | First head's attention reused in all layers (`skip=1`) | First layer gates normalized and reused by all layers |
| Train gate | `sigmoid((logit(U)+logAlpha)/0.66)` stretched by 1.2, shifted -0.1, hardtanh | `gate_weights(..., 'stochastic')` |
| Eval gate | `sigmoid(logAlpha/0.66)` stretched, shifted, clipped | `gate_weights(..., 'deterministic')` |
| Probability | `sigmoid(logAlpha - 0.66 * log(0.1/1.1))` | `l0_probability` |
| Sparsity penalty | `.sum()` of first head's probabilities, before loop gate override | `GateResult.l0_sum = probabilities.sum()` |
| Self loops | Add one loop per node, overwrite its attention with one | Explicit loops, force gates to one in message passing |
| Normalization | Gate sum over **destination**, no softmax in L0 mode | `index_add` normalization by destination sum |
| Attention dropout | After normalization | `attention_dropout` on normalized gates |
| Train objective | CE + λ L0 | CE + λ L0 **+ β soft squared-DP** (new) |
| Checkpoint | Validation accuracy, ties reset patience | Same by default; optional val_loss ablation |
| Evaluation | GAT deterministic gates on validation/test | SGAT deterministic gates; extra hard-binary diagnostic |

## Deliberate differences / reproducibility limits

1. **DGL -> PyTorch scatter** for modern Python/PyTorch support. Tensor operation ordering, random number streams, floating point rounding and execution runtime can differ.
2. Reference DGL graph can store directed edges; our graph generator produces *unique undirected edges*. We add both directed messages and loops; we report and save both **directed** and **undirected-union/both** edge ratios to avoid conflation with older GCN experiments.
3. Reference code builds stochastic gates for all attention heads but passes only the first head to the message computation/penalty. This port samples **only the used first-head gate**, which matches its distribution but changes the random-number consumption order.
4. Reference computes edges `.nonzero()` and uses DGL graph traversal to accelerate updates; this port keeps all candidate edges and performs multiplication by zero-gated attention. This is mathematically equivalent but **not** yet a runtime speedup evaluation.
5. `beta=0` uses SGAT's sparse L0 branch with no fairness penalty. It is **not** a dense softmax GAT; a separate dense GAT baseline is a different experiment.
6. We haven't benchmarked this implementation against old DGL SGAT on an official original dataset, so do not claim numerical equivalence or faithful reproduction of published metrics.

**Audit basis:** Official `gat.py` lines 15–35 (gates), 81–155 (attention/normalization), 156–202 (layer reuse); `train.py` lines 78–125, 129–147 (loops, optimizer/loss/checkpoint); README Cora/PPI commands.
