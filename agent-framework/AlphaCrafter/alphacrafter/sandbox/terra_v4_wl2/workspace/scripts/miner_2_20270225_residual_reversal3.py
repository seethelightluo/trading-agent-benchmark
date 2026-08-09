import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception:pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); fr=px.shift(-1)/px-1
# residual short-term reversal: remove contemporaneous cross-sectional median return
for look in [2,3,5]:
 raw=r.rolling(look).sum(); common=raw.median(axis=1); f=-(raw.sub(common,axis=0)); f=f.sub(f.median(axis=1),axis=0)
 vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('look',look,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(f.notna().sum().sum()/len(U)/len(f),4))
# artifact best candidate 3d
raw=r.rolling(3).sum(); f=-(raw.sub(raw.median(axis=1),axis=0)); f=f.sub(f.median(axis=1),axis=0)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_2_20270225_residual_reversal3.csv',index=False)
