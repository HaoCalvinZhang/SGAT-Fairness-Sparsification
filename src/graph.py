import json
from pathlib import Path
import numpy as np

def graph_name(n, degree, hs, hy, seed, label_signal=0.5, sensitive_signal=0.5, feature_noise=1.0, sensitive_ratio=0.8):
    return (f'n{n}_d{degree:g}_hs{hs:.2f}_hy{hy:.2f}_seed{seed}'
            f'_ay{label_signal:g}_as{sensitive_signal:g}_noise{feature_noise:g}_s0{int(round(sensitive_ratio*100))}')

def generate_graph(n, degree, hs, hy, feature_dim, seed, label_signal=0.5, sensitive_signal=0.5, feature_noise=1.0, sensitive_ratio=0.8):
    """Four (s,y) blocks; P(S=0)=sensitive_ratio and P(Y=0|S)=0.5.
    Edge mixing has target same-S and same-Y proportions hs and hy.
    """
    if n < 8 or not (0 < hs < 1 and 0 < hy < 1) or degree <= 0 or degree >= n:
        raise ValueError('Require n>=8, 0<hs,hy<1 and 0<degree<n')
    if not (0 < sensitive_ratio < 1):
        raise ValueError('sensitive_ratio must be between 0 and 1')
    if hs < abs(2*sensitive_ratio-1):
        raise ValueError('Requested sensitive homophily is infeasible for this group ratio while preserving group endpoint proportions; increase --hs')
    if feature_dim != 2:
        raise ValueError('This interpretable feature model requires --feature-dim 2')
    if label_signal < 0 or sensitive_signal < 0 or feature_noise <= 0:
        raise ValueError('Signals must be >= 0 and feature noise > 0')
    rng = np.random.default_rng(seed)
    # Allocate exact 80/20 sensitive groups (for n=10000) and balanced labels
    # within each sensitive group, keeping S and Y independent.
    n_s0 = int(round(n*sensitive_ratio))
    n_s1 = n-n_s0
    if min(n_s0,n_s1) < 2:
        raise ValueError('Each sensitive group must have at least 2 nodes')
    counts_target = [n_s0//2, n_s0-n_s0//2, n_s1//2, n_s1-n_s1//2]
    blocks = np.concatenate([np.full(c,b,dtype=np.int8) for b,c in enumerate(counts_target)])
    rng.shuffle(blocks)
    s = blocks // 2
    y = blocks % 2
    members = [np.flatnonzero(blocks == b) for b in range(4)]
    counts = np.array([len(x) for x in members], dtype=np.int64)

    # Undirected edge mass by sensitive groups. Off-diagonal entries
    # represent ordered endpoint mass; matrix sums to one.
    p0 = n_s0/n
    cross = (1-hs)/2
    ms = np.array([[p0-cross, cross],
                   [cross, 1-p0-cross]], dtype=float)
    # Label groups are balanced, so this matrix gives target label homophily.
    my = np.array([[hy/2, (1-hy)/2],
                   [(1-hy)/2, hy/2]], dtype=float)
    if np.any(ms < -1e-12):
        raise ValueError('Infeasible sensitive homophily / group ratio')
    pairs, capacity, probabilities = [], [], []
    target_edges = n*degree/2
    for a in range(4):
        for b in range(a,4):
            cap = counts[a]*(counts[a]-1)//2 if a==b else counts[a]*counts[b]
            mass = ms[a//2,b//2]*my[a%2,b%2]
            if a!=b:
                mass *= 2
            p_edge = target_edges*mass/cap if cap else 0
            pairs.append((a,b))
            capacity.append(cap)
            probabilities.append(p_edge)
    probabilities = np.asarray(probabilities,dtype=float)
    if np.max(probabilities)>1:
        raise ValueError('SBM probabilities exceed 1; lower average degree or adjust homophily')
    all_u, all_v = [], []
    for (a,b), cap, p in zip(pairs,capacity,probabilities):
        m = int(rng.binomial(int(cap),float(p)))
        if not m: continue
        # Sampling unique edge indices without allocating dense adjacency matrix.
        idx = rng.choice(int(cap),size=m,replace=False,shuffle=False)
        if a==b:
            k = counts[a]
            offsets = np.arange(k, dtype=np.int64)*(2*k-np.arange(k, dtype=np.int64)-1)//2
            row = np.searchsorted(offsets,idx,side='right')-1
            col = row+1+(idx-offsets[row])
            u,v=members[a][row],members[a][col]
        else:
            u,v=members[a][idx//counts[b]],members[b][idx%counts[b]]
        all_u.append(np.minimum(u,v)); all_v.append(np.maximum(u,v))
    u=np.concatenate(all_u).astype(np.int32) if all_u else np.array([],dtype=np.int32)
    v=np.concatenate(all_v).astype(np.int32) if all_v else np.array([],dtype=np.int32)
    # Independent RNG ensures that changing feature parameters does not change
    # the SBM topology, labels, or feature noise for the same graph seed.
    feature_rng = np.random.default_rng(np.random.SeedSequence([seed, 20261014]))
    noise = feature_rng.standard_normal((n, 2))
    x = np.empty((n, 2), dtype=np.float32)
    x[:, 0] = label_signal * (2*y - 1) + feature_noise * noise[:, 0]
    x[:, 1] = sensitive_signal * (2*s - 1) + feature_noise * noise[:, 1]
    # Intentionally NO per-feature z-score normalization: this preserves
    # the exact alpha / sigma separation and analytic feature-only baseline.
    return u,v,s.astype(np.int8),y.astype(np.int8),x

def actual_stats(u,v,s,y,n):
    return dict(nodes=n,edges=len(u),average_degree=2*len(u)/n,
                sensitive_homophily=float(np.mean(s[u]==s[v])) if len(u) else None,
                label_homophily=float(np.mean(y[u]==y[v])) if len(u) else None,
                sensitive_group_0_count=int(np.sum(s==0)), sensitive_group_1_count=int(np.sum(s==1)),
                sensitive_group_0_pct=float(np.mean(s==0)*100), sensitive_group_1_pct=float(np.mean(s==1)*100))

def split_indices(s,y,seed):
    rng=np.random.default_rng(seed)
    out=[[],[],[]]
    for b in range(4):
        ids=np.flatnonzero((s*2+y)==b)
        rng.shuffle(ids)
        a=int(.6*len(ids)); c=int(.8*len(ids))
        for k,part in enumerate((ids[:a],ids[a:c],ids[c:])):out[k].extend(part.tolist())
    return [np.asarray(z,dtype=np.int64) for z in out]

def save_original(path,u,v,s,y,x,split,meta):
    path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'graph.npz',u=u,v=v,s=s,y=y,x=x,train=split[0],val=split[1],test=split[2])
    (path/'metadata.json').write_text(json.dumps(meta,indent=2))


def load_graph(folder):
    with np.load(Path(folder)/'graph.npz', allow_pickle=False) as d:
        return {k:d[k] for k in ('u','v','s','y','x','train','val','test')}
