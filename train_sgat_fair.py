#!/usr/bin/env python3
"""SGAT official-mechanism PyTorch port + optional squared soft-DP training objective.

beta=0 is the L0-SGAT baseline, beta>0 adds ONLY fairness training loss.
The ORIGINAL first-head, feature-based attention, Hard Concrete, summed L0,
shared adjacency across layers, gate-sum normalization, and validation-accuracy
checkpoint protocol are preserved. No GCN replacement or fixed Top-k budget.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sgat_fair.model import SGAT, build_directed_edges
from sgat_fair.metrics import soft_dp_loss, classification_scores
from src.graph import load_graph
from plot_sgat_fair import summarize, create_plots


COLUMNS = [
    'graph', 'config_id', 'method', 'lambda_l0', 'beta', 'run_seed',
    'nodes', 'original_undirected_edges', 'original_directed_edges',
    'actual_keep_pct', 'expected_keep_pct', 'undirected_union_keep_pct',
    'undirected_both_keep_pct', 'actual_directed_edges',
    'test_accuracy_pct', 'test_dp_gap', 'test_eo_gap', 'test_fpr_gap',
    'test_equalized_odds_gap', 'test_worst_group_accuracy_pct',
    'hard_accuracy_pct', 'hard_dp_gap', 'hard_eo_gap',
    'train_ce_at_best', 'train_soft_dp_at_best', 'l0_sum_at_best',
    'weighted_l0_at_best', 'weighted_dp_at_best',
    'best_epoch', 'best_val_accuracy_pct', 'attention_grad_norm',
    'edge_file', 'seconds',
]


def pick_device(arg):
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arg == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use --device cpu')
    return torch.device(arg)


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if device.type == 'mps':
        torch.mps.manual_seed(seed)


def num(v):
    return f'{v:.9g}'.replace('-', 'minus').replace('.', 'p').replace('+', '')


def train_one(data, edges, root, args, lam, beta, seed, config_id, device, outdir):
    start = time.monotonic()
    seed_all(seed, device)
    x, y, s, train, val, test = data
    n = len(y)
    m_undirected = (edges.shape[1] - n) // 2
    model = SGAT(dim_in=x.shape[1], num_classes=2, hidden=args.hidden,
                 num_heads=args.num_heads, num_layers=args.num_layers,
                 out_heads=args.out_heads,
                 input_dropout=args.idrop, attention_dropout=args.adrop,
                 bias_l0=args.bias, temperature=args.temperature,
                 residual=args.residual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_epoch = 0
    best_val_acc = -1.0
    best_val_ce = float('inf')
    stale = 0
    last_grad = float('nan')
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, gates = model(x, edges, mode='stochastic', return_gates=True)
        ce = F.cross_entropy(logits[train], y[train])
        dp = soft_dp_loss(logits, s, train)
        loss = ce + lam * gates.l0_sum + beta * dp
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss lambda={lam}, beta={beta}, seed={seed}, epoch={epoch}')
        loss.backward()
        alpha_grad = model.layers[0].att_src.grad
        if alpha_grad is None or not torch.isfinite(alpha_grad).all():
            raise RuntimeError('Gradients failed to reach first-layer attention scoring')
        last_grad = float(alpha_grad.norm().item())
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, val_gates = model(x, edges, mode='deterministic', return_gates=True)
            val_ce = float(F.cross_entropy(val_logits[val], y[val]).item())
            val_acc = float((val_logits[val].argmax(-1) == y[val]).float().mean().item())
        # Reference SGAT updates checkpoint on equal validation accuracies.
        is_better = (val_acc >= best_val_acc - 1e-10) if args.checkpoint == 'val_accuracy' else (val_ce < best_val_ce - 1e-8)
        if is_better:
            best_val_acc = val_acc
            best_val_ce = val_ce
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if args.log_every and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f'  epoch={epoch:3d} lambda={lam:g} beta={beta:g} seed={seed} '
                  f'CE={ce.item():.5f} L0sum={gates.l0_sum.item():.1f} '
                  f'(lambda*L0={lam*gates.l0_sum.item():.4f}) '
                  f'DPsoft={dp.item():.7f} (beta*DP={beta*dp.item():.5f}) '
                  f'expected_keep={val_gates.expected_keep_pct.item():.2f}% '
                  f'actual_keep={val_gates.deterministic_keep_pct.item():.2f}% '
                  f'valacc={100*val_acc:.2f}%', flush=True)
        if args.patience > 0 and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError('No checkpoint was selected')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out, gates = model(x, edges, mode='deterministic', return_gates=True)
        hard_out, _ = model(x, edges, mode='binary', return_gates=False)
        measured = classification_scores(out, y, s, test)
        hard_measured = classification_scores(hard_out, y, s, test)
        best_ce = float(F.cross_entropy(out[train], y[train]).item())
        best_dp = float(soft_dp_loss(out, s, train).item())
        probs = gates.nonzero_probabilities[:2*m_undirected].cpu().numpy()
        gate_vals = gates.gate[:2*m_undirected].cpu().numpy()
        keep = gate_vals > 0
        union = keep[:m_undirected] | keep[m_undirected:]
        both = keep[:m_undirected] & keep[m_undirected:]
        source = edges[0, :2*m_undirected].cpu().numpy()
        dest = edges[1, :2*m_undirected].cpu().numpy()
        l0_sum = float(gates.l0_sum.item())
        expected = 100 * float(probs.mean())
        actual = 100 * float(keep.mean())

    edges_dir = outdir / 'selected_edges'
    edges_dir.mkdir(parents=True, exist_ok=True)
    edge_file = edges_dir / f'lambda_{num(lam)}_beta_{num(beta)}_seed_{seed}.npz'
    np.savez_compressed(edge_file,
                        src=source[keep], dst=dest[keep],
                        original_directed_indices=np.flatnonzero(keep),
                        undirected_union_indices=np.flatnonzero(union),
                        undirected_both_indices=np.flatnonzero(both),
                        lambda_l0=lam, beta=beta, seed=seed,
                        original_directed_edges=2*m_undirected)
    if args.save_gates:
        np.savez_compressed(edges_dir / f'gates_lambda_{num(lam)}_beta_{num(beta)}_seed_{seed}.npz',
                            src=source, dst=dest, log_alpha=gates.logits_alpha[:2*m_undirected].cpu().numpy(),
                            p_nonzero=probs, deterministic_gate=gate_vals)
    if args.save_checkpoints:
        ckpt_dir = outdir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {k: v.cpu() for k,v in best_state.items()},
                    'epoch': best_epoch, 'lambda_l0': lam, 'beta': beta, 'seed': seed,
                    'args': vars(args)}, ckpt_dir / f'lambda_{num(lam)}_beta_{num(beta)}_seed_{seed}.pt')

    return {
        'graph': root.name, 'config_id': config_id,
        'method': 'sgat_l0_original' if beta == 0 else 'sgat_l0_dp',
        'lambda_l0': lam, 'beta': beta, 'run_seed': seed,
        'nodes': n, 'original_undirected_edges': m_undirected,
        'original_directed_edges': 2*m_undirected,
        'actual_keep_pct': actual, 'expected_keep_pct': expected,
        'undirected_union_keep_pct': 100*float(union.mean()),
        'undirected_both_keep_pct': 100*float(both.mean()),
        'actual_directed_edges': int(keep.sum()),
        'test_accuracy_pct': measured['accuracy_pct'],
        'test_dp_gap': measured['dp_gap'], 'test_eo_gap': measured['eo_gap'],
        'test_fpr_gap': measured['fpr_gap'],
        'test_equalized_odds_gap': measured['equalized_odds_gap'],
        'test_worst_group_accuracy_pct': measured['worst_group_accuracy_pct'],
        'hard_accuracy_pct': hard_measured['accuracy_pct'],
        'hard_dp_gap': hard_measured['dp_gap'], 'hard_eo_gap': hard_measured['eo_gap'],
        'train_ce_at_best': best_ce, 'train_soft_dp_at_best': best_dp,
        'l0_sum_at_best': l0_sum, 'weighted_l0_at_best': lam * l0_sum,
        'weighted_dp_at_best': beta * best_dp,
        'best_epoch': best_epoch, 'best_val_accuracy_pct': 100*best_val_acc,
        'attention_grad_norm': last_grad,
        'edge_file': str(edge_file), 'seconds': round(time.monotonic()-start,2),
    }


def dataframe(csv_file):
    if not csv_file.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(csv_file)
    if df.empty: return pd.DataFrame(columns=COLUMNS)
    return df.drop_duplicates(subset=['lambda_l0','beta','run_seed'], keep='last')


def save_row(csv_file: Path, row: dict):
    # Atomic replace protects prior runs against interruption while writing.
    df = dataframe(csv_file)
    df = pd.DataFrame([row], columns=COLUMNS) if df.empty else pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.drop_duplicates(subset=['lambda_l0','beta','run_seed'],keep='last')
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    temp = csv_file.with_suffix('.csv.tmp')
    df.to_csv(temp, index=False, columns=COLUMNS)
    temp.replace(csv_file)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--graph', required=True, help='Dataset path containing original/graph.npz')
    p.add_argument('--lambdas', type=float, nargs='+', default=[0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5],
                   help='Original SGAT convention: lambda multiplies SUM of nonzero probabilities')
    p.add_argument('--betas', type=float, nargs='+', default=[0, 1, 10], help='Fairness weights; beta=0 is SGAT baseline')
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--train-seed', type=int, default=1042)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num-heads', type=int, default=2)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--out-heads', type=int, default=1)
    p.add_argument('--idrop', type=float, default=0.4)
    p.add_argument('--adrop', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--bias', type=float, default=0.0)
    p.add_argument('--temperature', type=float, default=.66)
    p.add_argument('--residual', action='store_true')
    p.add_argument('--checkpoint', choices=['val_accuracy','val_loss'], default='val_accuracy',
                   help='val_accuracy reproduces SGAT original checkpoint selection')
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--output-base', default='outputs', help='New independent results root')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--save-gates', action='store_true', help='Save nonzero probabilities and logits for diagnostics')
    p.add_argument('--save-checkpoints', action='store_true')
    p.add_argument('--plots-only', action='store_true')
    p.add_argument('--force', action='store_true', help='Recompute existing combinations; replace matching CSV rows')
    args = p.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in args.lambdas + args.betas) or not args.lambdas or not args.betas:
        p.error('lambda/beta must be finite and >=0')
    if not (args.runs>=1 and args.epochs>=1 and args.hidden>=1 and args.patience>=0
            and 0<=args.idrop<1 and 0<=args.adrop<1 and args.temperature>0):
        p.error('Invalid epochs/runs/patience/layers/dropout/temperature')
    root = Path(args.graph)
    graph_file = root / 'original' / 'graph.npz'
    if not graph_file.exists():
        p.error(f'Missing dataset: {graph_file}. Run generate.py first!')
    # These settings define an experiment identity; --runs and lambda/beta are deliberately
    # excluded, enabling a 1-seed sweep to resume as a 5-seed sweep.
    config = {k:getattr(args,k) for k in ('epochs','patience','hidden','num_heads','num_layers',
              'out_heads','idrop','adrop','lr','weight_decay','bias','temperature','residual','checkpoint')}
    config_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:10]
    out = Path(args.output_base) / root.name / f'config_{config_id}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps({'model':config, 'graph':str(graph_file),
        'loss':'train_CE + lambda * SUM(nonzero_probabilities including loops) + beta * squared_soft_DP_on_train',
        'source':'https://github.com/Yangyeeee/SGAT',
        'beta0_is_original_sgat_baseline':True},indent=2))
    csv_file = out/'runs.csv'
    if not args.plots_only:
        d = load_graph(root/'original')
        u, v = d['u'],d['v']
        n = len(d['y'])
        if len(u) < 1 or np.any(u>=v) or np.any(u<0) or np.any(v>=n):
            p.error('graph.npz must contain unique undirected original edges u<v')
        device = pick_device(args.device)
        x = torch.as_tensor(d['x'], dtype=torch.float32, device=device)
        y = torch.as_tensor(d['y'], dtype=torch.long, device=device)
        s = torch.as_tensor(d['s'], dtype=torch.long, device=device)
        train,val,test = (torch.as_tensor(d[k],dtype=torch.long,device=device) for k in ('train','val','test'))
        edges = build_directed_edges(u,v,n,device)
        data = x,y,s,train,val,test
        lambdas=sorted(set(args.lambdas)); betas=sorted(set(args.betas))
        seeds=list(range(args.train_seed,args.train_seed+args.runs))
        print(f'Graph={root.name} | N={n} | undirected E={len(u)} | directed E={2*len(u)} (excluding loops)')
        print(f'Model=original-SGAT-style GAT + L0 + beta*DP | device={device} | config={config_id}')
        print(f'Lambdas={lambdas} | Betas={betas} | Seeds={seeds}\n',flush=True)
        done=dataframe(csv_file)
        for lam in lambdas:
            for beta in betas:
                for seed in seeds:
                    key = ((done['lambda_l0'] == lam) & (done['beta'] == beta) & (done['run_seed'] == seed)) if len(done) else []
                    if len(done) and key.any() and not args.force:
                        print(f'[SKIP] lambda={lam:g} beta={beta:g} seed={seed}: result already saved')
                        continue
                    print(f'\n[START] lambda={lam:g} beta={beta:g} seed={seed}',flush=True)
                    row=train_one(data,edges,root,args,lam,beta,seed,config_id,device,out)
                    save_row(csv_file,row)
                    done=dataframe(csv_file)
                    print(f'[DONE] λ={lam:g} β={beta:g} seed={seed} | '
                          f'actual_keep={row["actual_keep_pct"]:.2f}% '
                          f'expected_keep={row["expected_keep_pct"]:.2f}% | '
                          f'ACC={row["test_accuracy_pct"]:.2f}% DP={row["test_dp_gap"]:.4f} '
                          f'EO={row["test_eo_gap"]:.4f} | '
                          f'hard_ACC={row["hard_accuracy_pct"]:.2f}% '
                          f'hard_DP={row["hard_dp_gap"]:.4f} '
                          f'epoch={row["best_epoch"]} secs={row["seconds"]:.1f}',flush=True)
    df=dataframe(csv_file)
    if df.empty:
        raise SystemExit('No saved runs yet; omit --plots-only on first execution')
    summary=summarize(df)
    summary.to_csv(out/'summary.csv',index=False)
    count=create_plots(df,summary,out/'plots')
    print('\nFINAL RESULTS (each group: mean ± sample SD; primary is deterministic SGAT TEST)')
    print(f'{"lambda":>12} {"beta":>8} {"n":>3} {"Actual Keep %":>18} {"Accuracy %":>18} {"DP Gap":>17} {"EO Gap":>17}')
    for _,r in summary.iterrows():
        fmt=lambda mean,std,prec: f'{mean:.{prec}f} +/- {std:.{prec}f}' if pd.notna(std) else f'{mean:.{prec}f} (SD n/a)'
        print(f'{r.lambda_l0:>12.3g} {r.beta:>8g} {int(r.n):>3d} '
              f'{fmt(r.actual_keep_mean,r.actual_keep_std,2):>18} '
              f'{fmt(r.accuracy_mean,r.accuracy_std,2):>18} '
              f'{fmt(r.dp_mean,r.dp_std,4):>17} '
              f'{fmt(r.eo_mean,r.eo_std,4):>17}')
    print(f'\nSaved {count} plots -> {out/"plots"}')
    print(f'Saved individual runs -> {csv_file}')
    print(f'Saved mean/std summary -> {out/"summary.csv"}',flush=True)


if __name__ == '__main__':
    main()
