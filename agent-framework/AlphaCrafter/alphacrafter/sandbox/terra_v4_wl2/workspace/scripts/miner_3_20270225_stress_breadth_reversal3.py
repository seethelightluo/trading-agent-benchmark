import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
EQ=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
D={s:get(s) for s in U}; C=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None}).sort_index()
v=get('VIX').set_index('date')['close'].reindex(C.index).ffill(); R=C.pct_change()
vshock=(v.pct_change()>v.pct_change().rolling(60,min_periods=30).quantile(.75)).shift(1)
breadth=(R[EQ].gt(0).mean(axis=1)<.25).shift(1)
# lagged, stress-conditioned short-term reversal, centered cross-section
f=(-R.rolling(3).sum()).where(vshock & breadth); f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; ic=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(ic); print('H',h,'dates',len(a),'avg_n',np.mean(ns) if ns else 0,'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
print('active',int((vshock&breadth).sum()),'value_coverage',f.notna().mean().mean(),'date_coverage',f.notna().any(axis=1).mean())
print('regimes')
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-25')]:
 y=C.shift(-1)/C-1; q=[]
 for dt in f.index:
  if str(dt)>=lo and str(dt)<=hi:
   z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lo,len(q),np.mean(q) if q else np.nan)
f.stack().rename('signal').reset_index().to_csv('../persistent/factor_signals_miner_3_20270225_stress_breadth_reversal3.csv',index=False)
