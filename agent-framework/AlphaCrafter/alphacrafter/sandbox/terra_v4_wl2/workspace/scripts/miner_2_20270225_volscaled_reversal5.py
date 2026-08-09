import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std();
# volatility-normalized medium-term reversal, lagged one session; cross-sectional rank-compatible
f=(-(r.rolling(5).sum()/vol)).shift(1)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avg_n',round(float(np.mean(ns)),2),'IC',round(float(np.mean(a)),8),'ICIR',round(float(np.mean(a)/np.std(a,ddof=1)),8),'hit',round(float(np.mean(a>0)),4))
# regimes
fr=px.shift(-1)/px-1; a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[x for d,x in a if str(d)>=lo and str(d)<=hi]; print(lab,'dates',len(q),'IC',round(float(np.mean(q)),8) if q else np.nan)
print('coverage',float(f.notna().mean().mean()),'turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_volscaled_reversal5.csv',index=False)
print('artifact rows',len(out))
