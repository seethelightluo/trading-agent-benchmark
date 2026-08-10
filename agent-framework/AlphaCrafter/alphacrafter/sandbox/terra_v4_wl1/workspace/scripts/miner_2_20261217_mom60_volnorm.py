import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change();
 # medium-horizon risk-adjusted momentum: 60d return divided by 20d realized vol, lagged
 d['factor']=(d.close.pct_change(60)/r.rolling(20).std()).shift(1)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); print('rows',len(x),'symbols',x.symbol.nunique())
for h in [1,5,10]:
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>=3: a.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(x.factor.notna().mean(),4)); print('turnover',round(x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4)); print('period',x.date.min().date(),x.date.max().date())
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>=3: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(reg=pd.cut(o.date,[pd.Timestamp('2019-12-31'),pd.Timestamp('2022-12-31'),pd.Timestamp('2024-12-31'),END],labels=['2020-22','2023-24','2025-26'])).groupby('reg',observed=True).ic.agg(['mean','count']).round(5).to_string())
