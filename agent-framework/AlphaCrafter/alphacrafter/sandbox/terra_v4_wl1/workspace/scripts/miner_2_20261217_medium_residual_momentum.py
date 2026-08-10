import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change()
 # Medium-term continuation, orthogonalized to short-term move: 20d return minus recent 3d return, lagged.
 d['factor']=(d.close.pct_change(20)-d.close.pct_change(3)).shift(1)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); print('period',x.date.min().date(),x.date.max().date(),'rows',len(x),'symbols',x.symbol.nunique())
for h in [1,5,10]:
 a=[]; ns=[]; dates=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: a.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g)); dates.append(dt)
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  z=pd.DataFrame({'date':dates,'ic':a}); print(z.assign(reg=np.select([z.date.dt.year<=2022,z.date.dt.year<=2024],["2020-22","2023-24"],default='2025-26')).groupby('reg').ic.agg(['mean','count']).round(6).to_string())
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4))
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
# save recoverable signal artifact
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261217_medium_residual_momentum_signal.csv',index=False)
