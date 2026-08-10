import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in(get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except:pass
p=pd.DataFrame({s:g(s).set_index('date').close for s in U}).sort_index();r=p.pct_change(); br=(r>0).mean(1); fr=p.shift(-1)/p-1
q=.10; w=5; st=(br<=br.rolling(60,min_periods=30).quantile(q)).shift(1)
f=(-r.rolling(w).sum()).where(st); f=f.sub(f.median(1),axis=0); a=[];ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
a=np.array(a); print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'active',int(st.sum()),'coverage',f.notna().sum().sum()/f.size)
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 vals=[]
 for d in f.index:
  if str(d)<lo or str(d)>hi:continue
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lab,len(vals),np.mean(vals) if vals else np.nan,np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_breadth_stress_reversal_q10_w5.csv',index=False)
