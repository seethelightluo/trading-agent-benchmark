import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END].copy(); d['ret']=d.close.pct_change();
 # Low-risk preference, lagged volatility and mildly conditioned on positive prior trend.
 vol=d.ret.rolling(20,min_periods=15).std().shift(1); trend=d.close.pct_change(20).shift(1)
 d['factor']=-(vol/(1+trend.clip(lower=-0.5,upper=0.5)))
 d['y']=d.close.shift(-1)/d.close-1; rows.append(d[['date','factor','y']].assign(symbol=s))
x=pd.concat(rows); obs=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna();
 if len(g)>=8:obs.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
a=np.array(obs); print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean());v=x.dropna();print('coverage',len(v)/sum(len(z) for z in rows));r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean());o=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8:o.append((dt,spearmanr(g.factor,g.y).statistic))
o=pd.DataFrame(o,columns=['date','ic']);print(o.assign(regime=pd.cut(o.date.dt.year,[2019,2022,2024,2027],labels=['2020-22','2023-24','2025-26'])).groupby('regime',observed=False).ic.agg(['mean','count']).to_string());x[['date','symbol','factor']].dropna().to_csv('scripts/miner_3_20261217_lowrisk_signal.csv',index=False)
