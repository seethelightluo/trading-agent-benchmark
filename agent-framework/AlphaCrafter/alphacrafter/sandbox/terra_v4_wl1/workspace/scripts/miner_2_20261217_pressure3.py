import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); r=d.close.pct_change(); d['vol']=r.rolling(20,min_periods=15).std().shift(1)
 pressure=((d.close-d.open)/(d.high-d.low).replace(0,np.nan)).clip(-1,1)
 d['factor']=-(pressure.rolling(3,min_periods=3).mean().shift(1))/(d.vol+1e-12)
 for h in (1,5,10): d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in (1,5,10):
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: a.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(a); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1):.8f} hit={(a>0).mean():.5f}')
v=x.dropna(subset=['factor']); print(f'coverage={len(v)/len(x):.6f}')
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print(f'turnover={r.diff().abs().mean(axis=1).mean():.8f}')
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(regime=np.select([o.date.dt.year<=2022,o.date.dt.year<=2024],["2020-22","2023-24"],default="2025-26")).groupby('regime').ic.agg(['mean','count']).round(8).to_string())
print(f'period={x.date.min().date()}..{x.date.max().date()} symbols={x.symbol.nunique()}')
# save recoverable signal artifact
x.to_csv('scripts/miner_2_20261217_pressure3_signal.csv',index=False)
