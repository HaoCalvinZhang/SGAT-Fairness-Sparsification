"""Binary node classification group fairness metrics."""
from __future__ import annotations
import numpy as np
import torch


def soft_dp_loss(logits: torch.Tensor, group: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # Only the specified split contributes to fairness training, not test labels.
    s = group[ids]
    p = logits[ids].softmax(dim=-1)[:, 1]
    p0, p1 = p[s == 0], p[s == 1]
    if p0.numel() == 0 or p1.numel() == 0:
        raise ValueError('Both sensitive groups must appear in fairness split')
    return (p0.mean() - p1.mean()).square()


def classification_scores(logits, labels, sensitive, ids):
    """Return test metrics (DP/EO computed on hard class decisions)."""
    pred = logits[ids].argmax(-1).detach().cpu().numpy()
    y = labels[ids].detach().cpu().numpy()
    s = sensitive[ids].detach().cpu().numpy()
    n0, n1 = int((s == 0).sum()), int((s == 1).sum())
    if not n0 or not n1:
        raise ValueError('Both sensitive groups required')
    g0, g1 = s == 0, s == 1
    pos_rates = [float((pred[g] == 1).mean()) for g in (g0, g1)]
    pos_ys = [g & (y == 1) for g in (g0, g1)]
    neg_ys = [g & (y == 0) for g in (g0, g1)]
    def rate(g, predicted):
        if g.sum() == 0: return float('nan')
        return float((pred[g] == predicted).mean())
    tpr = [rate(g, 1) for g in pos_ys]
    fpr = [rate(g, 1) for g in neg_ys]
    gap = lambda a,b: abs(a-b) if np.isfinite(a) and np.isfinite(b) else float('nan')
    group_acc = [float((pred[g] == y[g]).mean()) for g in (g0,g1)]
    eo = gap(tpr[0], tpr[1])
    fpr_gap = gap(fpr[0], fpr[1])
    return {
        'accuracy_pct': 100*float((pred==y).mean()),
        'dp_gap': gap(pos_rates[0],pos_rates[1]),
        'eo_gap': eo,
        'fpr_gap': fpr_gap,
        'equalized_odds_gap': max(eo, fpr_gap),
        'worst_group_accuracy_pct': 100*min(group_acc),
        'group0_accuracy_pct': 100*group_acc[0],
        'group1_accuracy_pct': 100*group_acc[1],
        'group0_positive_rate': pos_rates[0],
        'group1_positive_rate': pos_rates[1],
    }
