import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'mom10':d.close.pct_change(10),'vol20':r.rolling(20,min_periods=15).std(),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
p=x.pivot(index='date',columns='symbol',values='mom10'); med=p.median(axis=1); med[p.count(axis=1)<8]=np.nan
x['factor']=(x.mom10-x.date.map(med))/x.vol20.replace(0,np.nan)
def calc(z,h):
 a=[]; ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:
   q=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.asarray(a)
 return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for h in [1,5,10]:
 q=calc(x,h);print(f'H{h}: dates={q[0]} avgN={q[1]:.2f} IC={q[2]:.8f} ICIR={q[3]:.8f} hit={q[4]:.5f}')
for lo,hi,name in [('2020-01-01','2022-12-31','2020-22'),('2023-01-01','2024-12-31','2023-24'),('2025-01-01','2026-12-17','2025-26')]:
 q=calc(x[(x.date>=lo)&(x.date<=hi)],1); print(f'{name}: dates={q[0]} IC={q[2]:.8f} ICIR={q[3]:.8f}')
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print(f'coverage={len(v)/len(x):.6f} turnover={ranks.diff().abs().mean(axis=1).mean():.8f} rows={len(v)} symbols={x.symbol.nunique()} period={x.date.min().date()}..{x.date.max().date()}')
v[['date','symbol','factor']].to_csv('scripts/miner_2_20261218_residual_momentum_signal.csv',index=False)
