import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
breadth=(r.lt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1); active=breadth.ge(.50)
base=-r.rolling(3).sum(); f=base.where(active).sub(base.where(active).median(axis=1),axis=0); y=px.shift(-1)/px-1
obs=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: obs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in obs]); print('dates',len(a),'avgN',np.mean([x[2] for x in obs]),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'active',active.sum(),'coverage',f.notna().sum().sum()/(len(U)*len(f)))
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=np.array([v for d,v,n in obs if str(d)[:10]>=lo and str(d)[:10]<=hi]);print(name,'n',len(q),'IC',np.mean(q) if len(q) else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_breadth50_reversal3.csv',index=False)
