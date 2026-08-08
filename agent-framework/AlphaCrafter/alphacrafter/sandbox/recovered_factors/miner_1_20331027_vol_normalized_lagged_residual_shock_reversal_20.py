import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

root='../persistent/stock_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=[]
for a in assets:
    x=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
    px.append(x)
p=pd.concat(px,axis=1).sort_index().loc[:'2033-10-26']
r=p.pct_change()
res=r.sub(r.median(axis=1),axis=0)
# one completed-day residual shock scaled by entirely trailing own residual volatility; delay makes signal t available
scale=res.rolling(20,min_periods=15).std().shift(1).replace(0,np.nan)
factor=(-res.shift(1)/scale).replace([np.inf,-np.inf],np.nan)

def eval_h(h, subset=None):
    rows=[]
    fr=(p.shift(-h)/p-1)
    idx=factor.index if subset is None else factor.index.intersection(subset)
    for d in idx:
        z=pd.concat([factor.loc[d],fr.loc[d]],axis=1).dropna()
        if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=np.array(rows); return (q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0),len(q))
print('candidate: volatility_normalized_lagged_residual_shock_reversal_20')
print('visible_end',p.index.max().date(),'assets',len(assets),'factor_valid_cells',factor.notna().sum().sum(),'coverage',factor.notna().mean().mean())
for h in [1,5,10,20]: print('h',h,'IC ICIR hit n=',eval_h(h))
for name,start in [('2020_2025','2020-01-01'),('2026_2029','2026-01-01'),('2030_2033','2030-01-01'),('last12m','2032-10-26'),('last6m','2033-04-26')]:
 print(name, '1d',eval_h(1,pd.date_range(start,'2033-10-26')),'5d',eval_h(5,pd.date_range(start,'2033-10-26')))
# turnover based on daily Spearman ranks
turn=[]
for i in range(1,len(factor)):
 a,b=factor.iloc[i-1],factor.iloc[i]; both=a.notna()&b.notna()
 if both.sum()>=8: turn.append(1-spearmanr(a[both],b[both]).statistic)
print('turnover',np.mean(turn),'median_iqr',factor.quantile(.75,axis=1).sub(factor.quantile(.25,axis=1)).median(),'dates',factor.notna().sum(axis=1).ge(8).sum())
