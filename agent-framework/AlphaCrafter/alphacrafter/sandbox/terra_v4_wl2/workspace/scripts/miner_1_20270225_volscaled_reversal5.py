import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); f=-(r.rolling(5,min_periods=4).sum())/(r.rolling(20,min_periods=10).std()*np.sqrt(5))
# lag signal one completed day, forward next day return
fr=px.shift(-1)/px-1
ics=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
a=np.asarray(ics); print('dates',len(a),'range',dates[0],dates[-1],'avgN',np.mean(ns),'coverage',f.notna().sum().sum()/(len(U)*len(f)))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean([np.nan]))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-06-30'),('online','2026-07-01','2027-02-24')]:
 q=np.array([x for x,d in zip(a,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
 print(label,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
# rank turnover based on top/bottom rank changes
ranks=f.rank(axis=1,pct=True); print('rank_turnover',ranks.diff().abs().mean().mean())
# decay
for h in [1,3,5,10]:
 frh=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q),len(q))
# signal artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_reversal5.csv',index=False)
