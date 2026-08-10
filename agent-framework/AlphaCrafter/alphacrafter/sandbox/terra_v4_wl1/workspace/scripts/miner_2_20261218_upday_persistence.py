import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 # Trend quality: positive fraction of recent daily returns, lagged, isolates persistent advance from one-day jump.
 r=d.close.pct_change(); d['factor']=r.rolling(20,min_periods=15).apply(lambda x:(x>0).mean(),raw=True).shift(1)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1
 rows.append(d[['date','factor','y1','y5']].assign(symbol=s))
x=pd.concat(rows); print('period',x.date.min().date(),x.date.max().date(),'rows',len(x),'symbols',x.symbol.nunique())
for h in ['y1','y5']:
 a=[];ns=[];dates=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',h])
  if len(g)>=8:a.append(spearmanr(g.factor,g[h]).statistic);ns.append(len(g));dates.append(dt)
 a=np.array(a);print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h=='y1':
  z=pd.DataFrame({'date':dates,'ic':a});z['reg']=np.select([z.date.dt.year<=2022,z.date.dt.year<=2024],['2020-22','2023-24'],default='2025-26');print(z.groupby('reg').ic.agg(['mean','count']).round(6).to_string())
v=x.dropna(subset=['factor']);print('coverage',round(len(v)/len(x),4));q=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',round(q.diff().abs().mean(axis=1).mean(),4));x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261218_upday_persistence_signal.csv',index=False)
