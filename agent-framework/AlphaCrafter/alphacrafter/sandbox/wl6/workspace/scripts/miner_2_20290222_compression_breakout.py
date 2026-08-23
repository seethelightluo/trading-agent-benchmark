import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(s,days=3000)
   if x is not None and len(x): return x
  except: pass
raw={s:fetch(s) for s in U}; p=pd.DataFrame({s:x.set_index('date').close for s,x in raw.items() if x is not None}).sort_index(); r=p.pct_change()
# Breakout quality: medium-horizon return normalized by volatility, boosted when recent vol compresses versus its baseline.
ret=r.rolling(20,min_periods=15).sum(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
f=ret/(v20+1e-12)*(1-v20/(v60+1e-12))
for h in [1,5,10]:
 q=[]; ds=[]; ns=[]; fw=p.shift(-h)/p-1
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 q=pd.Series(q,index=pd.to_datetime(ds)); print(f'h={h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
  x=q[(q.index>=lo)&(q.index<=hi)]
  if len(x):print(f' {lo}-{hi}: n={len(x)} IC={x.mean():.6f}')
print('coverage',np.isfinite(f).sum().sum()/(f.shape[0]*f.shape[1]),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'instruments',p.shape[1],'dates',len(p))
