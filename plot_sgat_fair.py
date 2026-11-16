#!/usr/bin/env python3
"""Plot saved SGAT-versus-SGAT+DP runs without retraining."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    'actual_keep_pct': 'actual_keep',
    'expected_keep_pct': 'expected_keep',
    'test_accuracy_pct': 'accuracy',
    'test_dp_gap': 'dp',
    'test_eo_gap': 'eo',
    'hard_accuracy_pct': 'hard_accuracy',
    'hard_dp_gap': 'hard_dp',
}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lam,beta),g in runs.groupby(['lambda_l0','beta'],sort=True):
        row={'lambda_l0':float(lam),'beta':float(beta),'n':int(len(g))}
        for k,name in METRICS.items():
            v=pd.to_numeric(g[k],errors='coerce')
            row[name+'_mean']=float(v.mean())
            row[name+'_std']=float(v.std(ddof=1)) if v.count()>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def create_plots(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> int:
    if df.empty or summary.empty: return 0
    outdir.mkdir(parents=True,exist_ok=True)
    beta_values=sorted(df['beta'].unique())
    colors=plt.get_cmap('tab10').colors
    pal={b:colors[i%len(colors)] for i,b in enumerate(beta_values)}
    saved=0

    def save(fig,name):
        nonlocal saved
        fig.savefig(outdir/name,dpi=200,bbox_inches='tight')
        plt.close(fig)
        saved+=1

    # Parameter views: category labels avoids tiny lambda coefficients collapsing at x=0.
    lam_values=sorted(df['lambda_l0'].unique())
    x_labels=[f'{v:.3g}' for v in lam_values]
    index={v:i for i,v in enumerate(lam_values)}
    for col,ylabel,filename in [
        ('actual_keep','Actual directed Edge Keep (%)','lambda_vs_actual_keep.png'),
        ('expected_keep','Expected directed Edge Keep (%)','lambda_vs_expected_keep.png'),
        ('accuracy','Test Accuracy (%)','lambda_vs_accuracy.png'),
        ('dp','Test DP Gap (lower better)','lambda_vs_dp_gap.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for beta in beta_values:
            g=summary[summary.beta==beta].sort_values('lambda_l0')
            xs=[index[v] for v in g.lambda_l0]
            means=g[col+'_mean'].values
            ax.plot(xs,means,'-o',color=pal[beta],label=f'β={beta:g}',lw=1.5,ms=5)
            std=g[col+'_std'].to_numpy(dtype=float)
            if np.isfinite(std).any():
                ax.fill_between(xs,means-np.nan_to_num(std),means+np.nan_to_num(std),alpha=.10,color=pal[beta])
        ax.set_xticks(range(len(lam_values)),x_labels)
        ax.set_xlabel('L0 coefficient λ (original SGAT sum convention)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend(title='Fairness weight')
        fig.tight_layout()
        save(fig,filename)

    # Actual sparsity sweeps. Different betas may learn different edge counts.
    for col,ylabel,filename in [
        ('accuracy','Test Accuracy (%)','actual_keep_vs_accuracy.png'),
        ('dp','Test DP Gap','actual_keep_vs_dp_gap.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for beta in beta_values:
            g=summary[summary.beta==beta].sort_values('actual_keep_mean')
            ax.plot(g.actual_keep_mean,g[col+'_mean'],'-o',color=pal[beta],label=f'β={beta:g}',lw=1.5,ms=5)
        ax.set_xlabel('Actual directed Edge Keep (%)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        save(fig,filename)

    # Each seed + big stars for mean; point AREA encodes sparsity.
    fig,ax=plt.subplots(figsize=(8,6))
    for beta in beta_values:
        g=df[df.beta==beta]
        ax.scatter(g.test_dp_gap,g.test_accuracy_pct,
                   s=18+1.8*g.actual_keep_pct,color=pal[beta],alpha=.28,label=f'β={beta:g} (runs)')
        means=summary[summary.beta==beta]
        ax.scatter(means.dp_mean,means.accuracy_mean,
                   s=130+2.0*means.actual_keep_mean,marker='*',
                   c=[pal[beta]],edgecolors='#333333',linewidths=.6,zorder=5)
    ax.set_xlabel('Test DP Gap (lower better)')
    ax.set_ylabel('Test Accuracy (%) (higher better)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9,loc='best')
    ax.set_title('Color: β  |  Star: mean  |  Size: actual directed keep')
    fig.tight_layout()
    save(fig,'accuracy_dp_scatter_beta_size_keep.png')

    # Expected-vs-actual keep graph, diagnostic for threshold effects.
    fig,ax=plt.subplots(figsize=(7,6))
    for beta in beta_values:
        g=df[df.beta==beta]
        ax.scatter(g.expected_keep_pct,g.actual_keep_pct,s=45,alpha=.7,color=pal[beta],label=f'β={beta:g}')
    ax.plot([0,100],[0,100],'k--',alpha=.35,label='Expected = Actual')
    ax.set_xlim(0,102);ax.set_ylim(0,102)
    ax.set_xlabel('Expected directed Edge Keep (%)')
    ax.set_ylabel('Actual directed Edge Keep (%)')
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    save(fig,'expected_vs_actual_keep.png')

    # Lambda-beta heatmaps; missing cells labeled NaN rather than fabricated.
    for col,name,fmt in [('actual_keep','keep','%.1f'),('accuracy','accuracy','%.1f'),('dp','dp','%.3f')]:
        table=summary.pivot(index='lambda_l0',columns='beta',values=col+'_mean')
        a=table.to_numpy(dtype=float)
        fig,ax=plt.subplots(figsize=(max(6,len(table.columns)*1.0+2),max(4,len(table.index)*.5+2)))
        im=ax.imshow(a,aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(table.columns)),[f'{b:g}' for b in table.columns]);ax.set_xlabel('β')
        ax.set_yticks(range(len(table.index)),[f'{l:.3g}' for l in table.index]);ax.set_ylabel('λ (original sum convention)')
        ax.set_title(f'Average {col}')
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i,j]):ax.text(j,i,fmt%a[i,j],ha='center',va='center',fontsize=8,color='white' if a[i,j] < np.nanmedian(a) else 'black')
        fig.colorbar(im,ax=ax,shrink=.83)
        fig.tight_layout()
        save(fig,f'heatmap_{name}.png')

    return saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-csv',required=True,help='Path printed after training (outputs/.../runs.csv)')
    a=p.parse_args()
    fp=Path(a.runs_csv)
    if not fp.exists():p.error('Runs CSV does not exist; train_sgat_fair.py first')
    df=pd.read_csv(fp).drop_duplicates(['lambda_l0','beta','run_seed'],keep='last')
    summary=summarize(df)
    summary.to_csv(fp.parent/'summary.csv',index=False)
    count=create_plots(df,summary,fp.parent/'plots')
    print(f'Saved {count} plots in: {fp.parent/"plots"}')

if __name__=='__main__':main()
