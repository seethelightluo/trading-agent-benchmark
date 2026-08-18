import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,5000)
   if d is not None and len(d)>100:return d
  except: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
P=pd.DataFrame({s:d.set_index(pd.to_datetime(d.date)).close for s,d in D.items()}).sort_index().groupby(level=0).last().ffill()
R=P.pct_change(); eq=[s for s in ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX'] if s in R]; de=[s for s in ['XAU','US10Y','CN10Y'] if s in R]
# Defensive leadership gate: lagged equity breadth weak and defensive relative return positive.
breadth=R[eq].rolling(10,min_periods=7).mean().mean(axis=1)
deflead=(R[de].rolling(20,min_periods=12).mean().mean(axis=1)-R[eq].rolling(20,min_periods=12).mean().mean(axis=1))
gate=((breadth.shift(1)<0)&(deflead.shift(1)>0))
base=R.rolling(10,min_periods=8).sum().shift(1); res=base.sub(base.median(axis=1),axis=0); vol=R.rolling(30,min_periods=15).std().shift(1)
F=(-res/vol).where(gate, np.nan); F.index.name='date'; F.to_csv('scripts/miner_2_20330401_defensive_leadership_residual_signal.csv')
print('assets',len(D),'dates',len(P),'gate_dates',int(gate.sum()),'coverage',round(F.notna().sum(axis=1).where(gate).mean()/len(D),4))
for h in [1,3,5,10]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):vals.append(c);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2) if ns else 0,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
# split available history
for lo,hi in [('2020','2026-12-31'),('2027','2029-12-31'),('2030','2033-12-31')]:
 a=[]
 for i in range(len(P)-1):
  if not(lo<=str(P.index[i].date())<=hi):continue
  z=pd.concat([F.iloc[i],P.iloc[i+1].div(P.iloc[i])-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1]);
   if np.isfinite(c):a.append(c)
 a=np.array(a); print('REG',lo,'dates',len(a),'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
