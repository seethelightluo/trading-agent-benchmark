import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); d['ret']=d.close.pct_change()
 # Medium-horizon momentum, risk adjusted and lagged. Signal is deliberately distinct from short reversal.
 d['factor']=d.close.pct_change(20).shift(1)/(d.ret.rolling(60,min_periods=40).std().shift(1)+1e-12)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(obs); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
v=x.dropna(subset=['factor']); print('coverage=',len(v)/sum(len(z) for z in rows))
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover=',r.diff().abs().mean(axis=1).mean())
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(regime=pd.cut(o.date.dt.year,[2019,2022,2024,2026,2027],labels=['2020-22','2023-24','2025-26','end'])).groupby('regime',observed=False).ic.agg(['mean','count']).to_string())
print('period',x.date.min().date(),x.date.max().date(),'symbols',x.symbol.nunique())
# Save recoverable signal artifact.
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_3_20261217_medium_momentum_signal.csv',index=False)
