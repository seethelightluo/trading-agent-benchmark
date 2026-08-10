import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); d['r']=d.close.pct_change()
 rows.append(d[['date','close','r']].assign(symbol=s))
x=pd.concat(rows)
wide=x.pivot(index='date',columns='symbol',values='r'); disp=wide.std(axis=1); disp[wide.count(axis=1)<8]=np.nan
base=disp.rolling(60,min_periods=30).median(); scale=disp.rolling(60,min_periods=30).std()
amp=(1+(disp-base)/(scale+1e-12)).clip(0.25,2.5).shift(1).rename('amp')
x=x.join(amp,on='date'); x['factor']=(-x.groupby('symbol').close.pct_change(3).shift(1)*x.amp)
for h in [1,5,10]: x[f'y{h}']=x.groupby('symbol').close.shift(-h)/x.close-1
x[['date','symbol','factor']].dropna().to_csv('../persistent/index_data/miner_2_20261217_dispersion_tilt_reversal_signal.csv',index=False)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8: obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(obs); print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
print('coverage',x.factor.notna().mean(),'symbols',x.symbol.nunique(),'period',x.date.min().date(),x.date.max().date())
r=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic']); print(o.assign(reg=o.date.dt.year.map(lambda y:'2020-22' if y<=2022 else ('2023-24' if y<=2024 else '2025-26'))).groupby('reg').ic.agg(['mean','count']).round(6).to_string())
print('max_abs_library_correlation=not_computed')
