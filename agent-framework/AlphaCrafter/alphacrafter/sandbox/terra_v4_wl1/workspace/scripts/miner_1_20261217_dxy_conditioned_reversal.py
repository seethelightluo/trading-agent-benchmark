import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END][['date','close']].copy(); d['ret5']=d.close.pct_change(5)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 base.append(d.assign(symbol=s))
x=pd.concat(base)
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date')
macro=macro[macro.date<=END]; macro['dr20']=macro.close.pct_change(20)
m=macro[['date','dr20']].copy(); m['gate']=(m.dr20.shift(1)>0).astype(float)
x=x.merge(m,on='date',how='left')
med=x.groupby('date').ret5.transform('median')
x['factor']=-(x.ret5-med).shift(1)*x['gate']
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.array(obs); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
valid=x.dropna(subset=['factor']); print('coverage=',len(valid)/len(x),'symbols=',x.symbol.nunique(),'period=',x.date.min().date(),x.date.max().date())
r=valid.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover=',r.diff().abs().mean(axis=1).mean())
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>1: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(reg=np.where(o.date.dt.year<=2022,'2020-22',np.where(o.date.dt.year<=2024,'2023-24','2025-26'))).groupby('reg').ic.agg(['mean','count']).round(5).to_string())
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_1_20261217_dxy_conditioned_reversal_signal.csv',index=False)
