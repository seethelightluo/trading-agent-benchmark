import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change()
# Trend acceleration: recent 10d return relative to the preceding 20d return,
# normalized by trailing 20d volatility; all inputs lagged one day.
recent=r.rolling(10).sum(); prior=r.shift(10).rolling(20).sum(); vol=r.rolling(20).std()
f=((recent-prior)/vol.replace(0,np.nan)).shift(1)
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
for h in [1,5,10]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
  q=[v for d,v in zip(f.index,a) if str(d)>=lo and str(d)<=hi] # approximate not aligned due invalid dates
  if q: print(' ',lab,len(q),round(np.mean(q),6))
# artifact for possible admission
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_2_20270225_trend_accel_norm.csv',index=False)
print('coverage',round(f.notna().mean().mean(),4),'dates',len(f),'instruments',len(U))
