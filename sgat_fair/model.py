"""SGAT implementation in modern PyTorch, ported from the Ye/Ji DGL reference.

Reference: https://github.com/Yangyeeee/SGAT/blob/master/gat.py

Core reference behaviors preserved:
- L0 log-alpha comes from first GAT layer's projected node features and attention,
  plus a learnable scalar bias; *not* independent free parameters per edge.
- Concrete random gate in training, temperature-scaled deterministic gate in eval.
- One first-head gate shared across attention heads AND across all network layers.
- SGAT normalizes gate weights by their sum per destination (NOT edge_softmax).
- Self-loop gates are forced to one in message passing.
- Expected L0 regularizer is an unnormalized SUM, including self-loop logits,
  as in the reference first-layer loss computation.
- Edge list is directed; undirected input graph yields two directed messages.

Differences from 2019 code: tensor scatter ops replace old DGL graph APIs,
no removed .nonzero() edge-index pruning optimization in the forward pass,
Torch/Python versions are modern. Gradients and zero-gated messages are kept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class GateResult:
    l0_sum: Tensor
    expected_keep_pct: Tensor
    deterministic_keep_pct: Tensor
    normalized_weights: Tensor
    gate: Tensor
    logits_alpha: Tensor
    nonzero_probabilities: Tensor


class SparseAttentionLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, heads: int, feature_dropout: float,
                 attention_dropout: float, bias_l0: float = 0.0, residual: bool = False):
        super().__init__()
        self.heads, self.out_dim = heads, dim_out
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim_in, heads * dim_out, bias=False)
        # As in the author's DGL gat.py, attention vectors are broadcast over heads.
        self.att_src = nn.Parameter(torch.empty(1, 1, dim_out))
        self.att_dst = nn.Parameter(torch.empty(1, 1, dim_out))
        self.bias_l0 = nn.Parameter(torch.tensor(float(bias_l0)))
        self.residual = residual
        self.residual_projection = nn.Linear(dim_in, heads * dim_out, bias=False) if residual and dim_in != heads * dim_out else None
        nn.init.xavier_normal_(self.projection.weight, gain=math.sqrt(2.0))
        nn.init.xavier_normal_(self.att_src, gain=math.sqrt(2.0))
        nn.init.xavier_normal_(self.att_dst, gain=math.sqrt(2.0))
        if self.residual_projection is not None:
            nn.init.xavier_normal_(self.residual_projection.weight, gain=math.sqrt(2.0))

    def projected(self, x: Tensor) -> Tensor:
        return self.projection(self.feature_dropout(x)).reshape(x.shape[0], self.heads, self.out_dim)

    def gate_logits_first_head(self, projected: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        src_logits = (projected * self.att_src).sum(dim=-1)[:, 0]
        dst_logits = (projected * self.att_dst).sum(dim=-1)[:, 0]
        return src_logits[src] + dst_logits[dst] + self.bias_l0

    def aggregate(self, x: Tensor, projected: Tensor, src: Tensor, dst: Tensor, normalized_weights: Tensor) -> Tensor:
        edge_weight = self.attention_dropout(normalized_weights)
        messages = projected[src] * edge_weight[:, None, None]
        result = projected.new_zeros(projected.shape).index_add(0, dst, messages)
        if self.residual:
            if self.residual_projection is None:
                result = result + x.reshape(x.shape[0], self.heads, self.out_dim)
            else:
                result = result + self.residual_projection(x).reshape(x.shape[0], self.heads, self.out_dim)
        return result


class SGAT(nn.Module):
    def __init__(self, dim_in: int, num_classes: int = 2, hidden: int = 64,
                 num_heads: int = 2, num_layers: int = 1, out_heads: int = 1,
                 input_dropout: float = 0.4, attention_dropout: float = 0.3,
                 bias_l0: float = 0.0, temperature: float = 0.66,
                 gamma: float = -0.1, zeta: float = 1.1,
                 residual: bool = False):
        super().__init__()
        if num_layers < 1 or min(num_heads, out_heads, hidden) < 1:
            raise ValueError('num_layers/heads/hidden must be >= 1')
        if not (temperature > 0 and gamma < 0 and zeta > 1):
            raise ValueError('Temperature must be positive, gamma<0, zeta>1')
        self.temperature = float(temperature)
        self.gamma, self.zeta = float(gamma), float(zeta)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = dim_in if i == 0 else hidden * num_heads
            self.layers.append(SparseAttentionLayer(in_size, hidden, num_heads, input_dropout,
                                                    attention_dropout, bias_l0=bias_l0, residual=(residual and i > 0)))
        self.layers.append(SparseAttentionLayer(hidden * num_heads, num_classes, out_heads,
                                                input_dropout, attention_dropout,
                                                bias_l0=bias_l0, residual=residual))

    def l0_probability(self, log_alpha: Tensor) -> Tensor:
        return torch.sigmoid(log_alpha - self.temperature * math.log(-self.gamma / self.zeta))

    def gate_weights(self, log_alpha: Tensor, mode: str) -> Tensor:
        if mode == 'stochastic':
            u = torch.rand_like(log_alpha).clamp_(1e-7, 1-1e-7)
            logits = (torch.logit(u) + log_alpha) / self.temperature
        elif mode in ('deterministic', 'binary'):
            logits = log_alpha / self.temperature
        else:
            raise ValueError('mode must be stochastic, deterministic, or binary')
        gate = (torch.sigmoid(logits) * (self.zeta - self.gamma) + self.gamma).clamp(0, 1)
        if mode == 'binary':
            gate = (gate > 0).to(gate.dtype)
        return gate

    def forward(self, x: Tensor, edges: Tensor, *, mode: str | None = None,
                return_gates: bool = False) -> tuple[Tensor, GateResult | None]:
        """edges: [2, directed_edges + self_loops]; loops must be the final N positions.

        Returns (class logits [N,2], stats). `mode=None` uses train/eval default.
        """
        if mode is None:
            mode = 'stochastic' if self.training else 'deterministic'
        src, dst = edges[0], edges[1]
        n = x.shape[0]
        if edges.shape[1] < n:
            raise ValueError('Need an explicit self-loop for every node')
        # Data loader guarantees that its N terminal edges are self-loops.
        e_no_loops = edges.shape[1] - n
        ft = self.layers[0].projected(x)
        log_alpha = self.layers[0].gate_logits_first_head(ft, src, dst)
        p_nonzero = self.l0_probability(log_alpha)
        gate = self.gate_weights(log_alpha, mode)
        # Reference SGAT computes L0 penalty before replacing self-loop gates.
        gate = torch.cat((gate[:e_no_loops], torch.ones_like(gate[e_no_loops:])))
        # Normalize raw L0 gates by their sum per destination. NOT softmax.
        sums = gate.new_zeros(n).index_add(0, dst, gate)
        attn = gate / sums[dst].clamp_min(1e-12)
        h = F.elu(self.layers[0].aggregate(x, ft, src, dst, attn).flatten(1))
        for layer in self.layers[1:-1]:
            ft = layer.projected(h)
            h = F.elu(layer.aggregate(h, ft, src, dst, attn).flatten(1))
        output_layer = self.layers[-1]
        out_proj = output_layer.projected(h)
        outputs = output_layer.aggregate(h, out_proj, src, dst, attn).mean(dim=1)
        details = None
        if return_gates:
            details = GateResult(
                l0_sum=p_nonzero.sum(),
                expected_keep_pct=p_nonzero[:e_no_loops].mean() * 100,
                deterministic_keep_pct=(gate[:e_no_loops] > 0).float().mean() * 100,
                normalized_weights=attn,
                gate=gate,
                logits_alpha=log_alpha,
                nonzero_probabilities=p_nonzero,
            )
        return outputs, details


def build_directed_edges(u, v, num_nodes: int, device: torch.device) -> Tensor:
    """Add both directions of every input undirected edge, then all self-loops."""
    a = torch.as_tensor(u, dtype=torch.long, device=device)
    b = torch.as_tensor(v, dtype=torch.long, device=device)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError('u/v must be one-dimensional arrays with same length')
    ids = torch.arange(num_nodes, device=device)
    return torch.stack((torch.cat((a, b, ids)), torch.cat((b, a, ids))), dim=0)
