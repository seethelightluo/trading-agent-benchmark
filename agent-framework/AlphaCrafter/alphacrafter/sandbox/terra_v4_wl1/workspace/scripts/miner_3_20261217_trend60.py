import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['ret']=d.close.pct_change();
 # Trend persistence: lagged 60-day return, normalized by lagged 60-day realized risk.
 d['factor']=d.close.pct_change(60).shift(1)/(d.ret.rolling(60,min_periods=40).std().shift(1)+1e-12)
 d['y1']=d.close.shift(-1)/d.close-1; rows.append(d[['date','factor','y1']].assign(symbol=s))
x=pd.concat(rows); obs=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna();
 if len(g)>=8: obs.append(spearmanr(g.factor,g.y1).statistic); ns.append(len(g))
a=np.array(obs); print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
v=x.dropna(); print('coverage',len(v)/sum(len(z) for z in rows)); r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean()); o=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8:o.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(o,columns=['date','ic']); print(o.assign(regime=pd.cut(o.date.dt.year,[2019,2022,2024,2027],labels=['2020-22','2023-24','2025-26'])).groupby('regime',observed=False).ic.agg(['mean','count']).to_string()); print('period',x.date.min().date(),x.date.max().date()); x[['date','symbol','factor']].dropna().to_csv('scripts/miner_3_20261217_trend60_signal.csv',index=False)
