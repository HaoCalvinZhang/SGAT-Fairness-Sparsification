from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def summarize(df):
    rows=[]
    for (lam,beta),g in df.groupby(['lambda_l0','beta'],sort=True):
        row=dict(lambda_l0=float(lam),beta=float(beta),n=len(g))
        for c,nm in [('actual_keep_pct','actual_keep'),('expected_keep_pct','expected_keep'),('test_accuracy_pct','accuracy'),('hard_accuracy_pct','hard_accuracy'),('test_dp_gap','dp'),('test_eo_gap','eo'),('test_soft_dp_gap','soft_dp'),('test_pred_positive_rate','positive_rate')]:
            v=pd.to_numeric(g[c],errors='coerce')
            row[nm+'_mean']=v.mean();row[nm+'_std']=v.std(ddof=1) if len(v)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def create_plots(df,summary,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    count=0
    for x,y,fn in [('lambda_l0','actual_keep_mean','lambda_vs_keep.png'),('lambda_l0','dp_mean','lambda_vs_dp.png'),('lambda_l0','accuracy_mean','lambda_vs_accuracy.png'),('actual_keep_mean','accuracy_mean','keep_vs_accuracy.png'),('actual_keep_mean','dp_mean','keep_vs_dp.png'),('actual_keep_mean','soft_dp_mean','keep_vs_soft_dp.png'),('lambda_l0','positive_rate_mean','lambda_vs_positive_rate.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for beta,g in summary.groupby('beta'):
            g=g.sort_values(x)
            ax.plot(g[x],g[y],'-o',label=f'beta={beta:g}')
        ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        fig.savefig(outdir/fn,dpi=180);plt.close(fig);count+=1
    return count
