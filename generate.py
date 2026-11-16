#!/usr/bin/env python3
"""Stage 1: create or reuse an immutable original graph."""
import argparse, json
from pathlib import Path
from src.graph import graph_name, generate_graph, split_indices, save_original, actual_stats

p=argparse.ArgumentParser()
p.add_argument('--out',default='results')
p.add_argument('--n',type=int,default=10000)
p.add_argument('--degree',type=float,default=20)
p.add_argument('--hs',type=float,default=.8)
p.add_argument('--hy',type=float,default=.2)
p.add_argument('--feature-dim',type=int,default=2)
p.add_argument('--label-signal',type=float,default=0.5)
p.add_argument('--sensitive-signal',type=float,default=0.5)
p.add_argument('--feature-noise',type=float,default=1.0)
p.add_argument('--sensitive-ratio',type=float,default=0.8,help='Fraction of nodes in S=0 (default 0.8)')
p.add_argument('--graph-seed',type=int,default=42)
p.add_argument('--overwrite',action='store_true')
a=p.parse_args()
name=graph_name(a.n,a.degree,a.hs,a.hy,a.graph_seed,a.label_signal,a.sensitive_signal,a.feature_noise,a.sensitive_ratio)
folder=Path(a.out)/'datasets'/name/'original'
file=folder/'graph.npz'
if file.exists() and not a.overwrite:
    with __import__('numpy').load(file) as d:
        if d['x'].shape[1]!=a.feature_dim:
            p.error('Existing graph has different feature dimension; choose another output directory or --overwrite')
    print('Reusing:',file)
else:
    u,v,s,y,x=generate_graph(a.n,a.degree,a.hs,a.hy,a.feature_dim,a.graph_seed,a.label_signal,a.sensitive_signal,a.feature_noise,a.sensitive_ratio)
    split=split_indices(s,y,a.graph_seed+1)
    meta={'requested':vars(a),'actual':actual_stats(u,v,s,y,a.n),
          'generator':'four-block SBM with controlled sensitive ratio and independent label balance; calibrated homophily',
          'features':'x[:,0]=label_signal*(2*y-1)+feature_noise*eps0; x[:,1]=sensitive_signal*(2*s-1)+feature_noise*eps1; eps~N(0,1); no z-score',
          'feature_model':{'label_signal':a.label_signal,'sensitive_signal':a.sensitive_signal,'noise_std':a.feature_noise,'feature_dim':2},
          'edge_format':'undirected unique edges u<v, no self-loops',
          'split':'stratified (s,y), 60/20/20'}
    save_original(folder,u,v,s,y,x,split,meta)
    print('Saved:',file)
    print(json.dumps(meta['actual'],indent=2))
print('Graph ID:',name)
