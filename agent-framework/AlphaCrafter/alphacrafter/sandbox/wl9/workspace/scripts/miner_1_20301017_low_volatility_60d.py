import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date); q=q.drop_duplicates('date').set_index('date').sort_index(); data[s]=q.close.astype(float)
p=pd.DataFrame(data).sort_index(); r=p.pct_change()
# Defensive low-volatility: inverse 30d realized volatility, smoothed and lagged.
# Cross-sectional rank makes the signal comparable across asset classes.
rv=r.rolling(30,min_periods=20).std()
f=(-rv).rank(axis=1,pct=True).rolling(3,min_periods=2).mean().shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(252)
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
 if h==60:
  for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-12-31','2030YTD')]:
   q=a[(a.index>=lo)&(a.index<=hi)]; print(f' regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan:.6f}')
print(f'turnover_proxy={f.diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(p)}')
f.to_csv('scripts/miner_1_20301017_low_volatility_signal.csv',index_label='date')
