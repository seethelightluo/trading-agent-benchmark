import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x):return x
  except:pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); ret=px.pct_change()
# Short-term reversal scaled by each asset's trailing volatility, lagged one day.
f=(-ret.rolling(3).sum()/ret.rolling(20).std()).shift(1)
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
for h in [1,5,10]:
 vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt);ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),8),'ICIR',round(np.mean(a)/np.std(a,ddof=1),8),'hit',round(np.mean(a>0),4))
 if h==5:
  for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
   q=np.array([v for d,v in zip(dates,vals) if str(d)[:10]>=lo and str(d)[:10]<=hi]);print(name,'dates',len(q),'IC',round(q.mean(),8) if len(q) else np.nan,'ICIR',round(q.mean()/q.std(ddof=1),8) if len(q)>1 else np.nan)
print('coverage',round(f.notna().sum().sum()/(len(f)*len(U)),4),'instruments',len(U))
rank=f.rank(axis=1,pct=True); t=[]
for i in range(1,len(rank)):
 z=(rank.iloc[i]-rank.iloc[i-1]).dropna()
 if len(z)>=8:t.append(abs(z).mean())
print('turnover',round(np.mean(t),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_reversal3.csv',index=False)
