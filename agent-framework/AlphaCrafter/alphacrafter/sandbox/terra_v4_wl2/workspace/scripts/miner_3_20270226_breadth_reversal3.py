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
# Lagged broad cross-asset downside breadth; reversal signal is formed only after broad stress.
breadth=(r.lt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1)
active=(breadth>=0.60)
f=(-r.rolling(3).sum()).where(active); f=f.sub(f.median(axis=1),axis=0)
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
for h in [1,5,10]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=[x[1] for x in a if str(x[0])[:10]>=lo and str(x[0])[:10]<=hi]; q=np.asarray(q);print(name,'dates',len(q),'IC',np.mean(q) if len(q) else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('active dates',int(active.sum()),'total',len(active),'coverage',f.notna().sum().sum()/len(U)/len(f))
# rank turnover on consecutive active dates
ranks=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(ranks)):
 if active.iloc[i] and active.iloc[i-1]: turn.append(np.mean(abs(ranks.iloc[i]-ranks.iloc[i-1]).dropna()))
print('turnover',np.mean(turn) if turn else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270226_breadth_reversal3.csv',index=False)
