import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 p=f'../persistent/stock_data/{s}.csv'
 d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 d['r']=d.close.pct_change()
 # Volatility-scaled short-term reversal, all inputs lagged to avoid lookahead.
 d['factor']=-(d.close.pct_change(3).shift(1))/(d.r.rolling(20,min_periods=15).std().shift(1)+1e-12)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(obs); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/sum(len(z) for z in rows),4))
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(year=o.date.dt.year).groupby('year').ic.agg(['mean','count']).round(5).to_string())
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
