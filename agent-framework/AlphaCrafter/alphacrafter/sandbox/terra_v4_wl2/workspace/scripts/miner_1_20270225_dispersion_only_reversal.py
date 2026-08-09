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
disp=r.std(axis=1); high=(disp>disp.rolling(60,min_periods=30).quantile(.75)).shift(1)
# On the day after an unusually dispersed cross-asset session, buy the lagging assets.
f=(-r.rolling(3).sum()).where(high); f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns) if ns else 0,'IC',np.mean(a) if len(a) else np.nan,'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(px.shift(-1)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for lab,aa,bb in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[x for d,x in a if str(d)>=aa and str(d)<=bb];print(lab,len(q),np.mean(q) if q else np.nan)
print('active',int(high.sum()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_dispersion_only_reversal.csv',index=False)
