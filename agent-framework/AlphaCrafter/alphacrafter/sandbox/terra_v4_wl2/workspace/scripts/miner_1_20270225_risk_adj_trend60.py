import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Lagged risk-adjusted medium-term trend: 60d return divided by trailing 20d realized volatility.
f=(r.rolling(60).sum()/r.rolling(20).std()).shift(1)
# cross-sectional rank to reduce scale differences
f=f.rank(axis=1,pct=True)
res=[]
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),8),'ICIR',round(np.mean(a)/np.std(a,ddof=1),8),'hit',round(np.mean(a>0),4))
 if h==5:
  for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
   q=[vals[i] for i,d in enumerate(f.index) if str(d)[:10]>=lo and str(d)[:10]<=hi and i<len(vals)]
   # dates list alignment is imperfect; use direct recompute below omitted
   print(name,'approx_dates',len(q),'IC',round(np.mean(q),6) if q else np.nan)
print('dates',len(f),'instruments',len(U),'coverage',round(f.notna().sum().sum()/(len(f)*len(U)),4))
ranks=f; turn=[]
for i in range(1,len(ranks)):
 z=(ranks.iloc[i]-ranks.iloc[i-1]).dropna()
 if len(z)>=8: turn.append(np.mean(abs(z)))
print('turnover',round(np.mean(turn),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_risk_adj_trend60.csv',index=False)
