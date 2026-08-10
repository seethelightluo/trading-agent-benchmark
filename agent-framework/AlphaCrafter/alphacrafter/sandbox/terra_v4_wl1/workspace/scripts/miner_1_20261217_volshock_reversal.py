import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change()
 # Volatility-shock reversal: recent 5d return, inverted, amplified when current volatility is
 # elevated versus its own trailing baseline. All inputs shifted one day before scoring.
 vol=d.r.rolling(10,min_periods=8).std(); base=vol.rolling(60,min_periods=30).median()
 d['factor']=-(d.close.pct_change(5).shift(1))*(vol.shift(1)/(base.shift(1)+1e-12))
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1; d['y10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); print('symbols',x.symbol.nunique(),'period',x.date.min().date(),x.date.max().date())
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.array(obs); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4))
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(regime=pd.cut(o.date.dt.year,[2019,2022,2024,2027],labels=['2020-22','2023-24','2025-27'])).groupby('regime',observed=True).ic.agg(['mean','count']).round(6).to_string())
# save raw signal artifact for downstream deterministic audit
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_1_20261217_volshock_reversal_signal.csv',index=False)
