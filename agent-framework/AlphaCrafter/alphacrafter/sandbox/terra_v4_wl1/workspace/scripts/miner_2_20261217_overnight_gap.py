import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy()
 # Overnight gap fade, signal observed after prior close for next session.
 d['factor']=-(d.open/d.close.shift(1)-1).shift(1)
 for h in (1,5,10):d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in (1,5,10):
 a=[];ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8:
   z=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(z):a.append(z);ns.append(len(g))
 a=np.array(a);print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.8f} ICIR={a.mean()/a.std(ddof=1):.8f} hit={(a>0).mean():.5f}')
v=x.dropna(subset=['factor']);print(f'coverage={len(v)/len(x):.6f}');r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print(f'turnover={r.diff().abs().mean(axis=1).mean():.8f}')
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8:
  z=spearmanr(g.factor,g.y1).statistic
  if np.isfinite(z):obs.append((dt,z))
o=pd.DataFrame(obs,columns=['date','ic']);o['regime']=np.select([o.date.dt.year<=2022,o.date.dt.year<=2024],['2020-22','2023-24'],default='2025-26');print(o.groupby('regime').ic.agg(['mean','count']).round(8).to_string());print(f'period={x.date.min().date()}..{x.date.max().date()} symbols={x.symbol.nunique()}');x.to_csv('scripts/miner_2_20261217_overnight_gap_signal.csv',index=False)
