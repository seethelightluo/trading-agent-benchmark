import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
v=get('VIX').set_index('date')['close'].reindex(px.index).ffill(); vx=v.pct_change()
disp=r.std(axis=1)
# lagged continuous stress intensity; all inputs shifted before signal
vp=vx.rolling(60,min_periods=30).rank(pct=True).shift(1)
dp=disp.rolling(60,min_periods=30).rank(pct=True).shift(1)
# continuous cross-asset reversal, amplified only by joint stress, lagged returns
stress=(vp.clip(0,1)*dp.clip(0,1)).clip(0,1)
f=(-r.rolling(3).sum()).mul(stress,axis=0)
f=f.sub(f.mean(axis=1),axis=0)
fr=px.shift(-1)/px-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
a=np.array(ics)
print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for h in [5,10]:
 aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(px.shift(-h)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 aa=np.array(aa);print('H',h,'dates',len(aa),'IC',np.mean(aa),'ICIR',np.mean(aa)/np.std(aa,ddof=1))
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 # recompute clean
 q=[]
 for dt in f.index:
  if str(dt)>=lo and str(dt)<=hi:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lab,len(q),np.mean(q) if q else np.nan)
print('coverage',f.notna().mean().mean(),'active_nonzero',np.mean(stress>0))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_continuous_stress_reversal.csv',index=False)
