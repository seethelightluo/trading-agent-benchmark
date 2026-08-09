import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Observation-only VIX is used solely for residualization; no orders are implied.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date')
v=v[v.date<=END][['date','close']].rename(columns={'close':'vix'})
v['vr']=v.vix.pct_change()
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END][['date','close']].copy(); d['r']=d.close.pct_change()
 d=d.merge(v,on='date',how='left')
 # all rolling inputs shifted: signal at t uses information through t-1
 ar=d.r.rolling(20,min_periods=15).std().shift(1)
 beta=d.r.rolling(20,min_periods=15).cov(d.vr).shift(1)/(d.vr.rolling(20,min_periods=15).var().shift(1)+1e-12)
 r3=d.close.pct_change(3).shift(1)
 v3=d.vix.pct_change(3).shift(1)
 residual=r3-beta*v3
 d['factor']=-residual/(ar*np.sqrt(3)+1e-12)
 d['y1']=d.close.shift(-1)/d.close-1; d['y5']=d.close.shift(-5)/d.close-1; d['y10']=d.close.shift(-10)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(obs); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
vv=x.dropna(subset=['factor']); print('coverage',len(vv)/sum(len(z) for z in rows))
r=vv.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(regime=pd.cut(o.date,[pd.Timestamp('2019-12-31'),pd.Timestamp('2022-12-31'),pd.Timestamp('2024-12-31'),END],labels=['2020-22','2023-24','2025-26'])).groupby('regime',observed=True).ic.agg(['mean','count']).round(6).to_string())
print('period',x.date.min().date(),x.date.max().date(),'symbols',x.symbol.nunique())
# provenance artifact for deterministic audit
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261217_vix_residual_reversal_signal.csv',index=False)
