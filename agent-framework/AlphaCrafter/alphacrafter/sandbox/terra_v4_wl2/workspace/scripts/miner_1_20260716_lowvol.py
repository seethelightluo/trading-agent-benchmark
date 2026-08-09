import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# Defensive low-volatility factor: inverse trailing 20d realized volatility, with only prior-day data
vol=r.rolling(20,min_periods=15).std(); f=-vol
for h in [1,5,10]:
 y=r.rolling(h).sum().shift(-h+1); xs=[]; ns=[]; turns=[]; prev=None
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y));ns.append(len(q))
  z=f.iloc[i].rank(pct=True)
  if prev is not None: turns.append(np.nanmean(abs(z-prev)))
  prev=z
 x=np.array(xs); print('h',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/15,'turn',np.nanmean(turns))
# regimes by calendar broad halves
for label,mask in [('2020-22',p.index<'2023-01-01'),('2023-24',(p.index>='2023-01-01')&(p.index<'2025-01-01')),('2025+',p.index>='2025-01-01')]:
 x=[]
 for i in range(len(p)-1):
  if not mask[i]: continue
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:x.append(q.f.corr(q.y))
 print(label,'dates',len(x),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1))
