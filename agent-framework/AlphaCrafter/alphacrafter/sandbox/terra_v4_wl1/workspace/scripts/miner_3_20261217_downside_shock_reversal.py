import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=END].copy(); r=d.close.pct_change()
 # Asymmetric downside-shock reversal: only prior negative 3d moves, scaled by lagged vol.
 r3=d.close.pct_change(3).shift(1); vol=r.rolling(20,min_periods=15).std().shift(1)
 d['factor']=(-r3/(vol+1e-12)).where(r3<0,0.0)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1:
   obs.append(spearmanr(g.factor,g[f'y{h}']).statistic); ns.append(len(g))
 a=np.asarray(obs)
 print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
v=x.dropna(subset=['factor']); print(f'coverage={len(v)/len(x):.6f} dates={x.date.nunique()} instruments={x.symbol.nunique()}')
r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print(f'turnover={r.diff().abs().mean(axis=1).mean():.6f}')
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>1: obs.append((dt,spearmanr(g.factor,g.y1).statistic))
o=pd.DataFrame(obs,columns=['date','ic'])
for label,lo,hi in [('2020-22',2020,2022),('2023-24',2023,2024),('2025-26',2025,2026)]:
 a=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic
 print(f'{label} dates={len(a)} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f}')
# Save recoverable signal artifact
x[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_downside_shock_signal.csv',index=False)
