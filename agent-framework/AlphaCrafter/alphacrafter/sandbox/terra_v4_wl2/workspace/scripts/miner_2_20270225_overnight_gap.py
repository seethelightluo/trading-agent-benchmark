import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
D={s:g(s).set_index('date') for s in U}; op=pd.DataFrame({s:D[s]['open'] for s in U}).sort_index(); cl=pd.DataFrame({s:D[s]['close'] for s in U}).reindex(op.index)
# Overnight gap, normalized by prior 20d close volatility; reversal predicts next close return.
gap=op/cl.shift(1)-1; vol=cl.pct_change().rolling(20).std();
for norm in ['raw','volscaled']:
 x=gap if norm=='raw' else gap/vol
 f=-x; f=f.sub(f.median(axis=1),axis=0); fr=cl.shift(-1)/cl-1
 vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals); print(norm,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(f.notna().sum().sum()/len(U)/len(f),4))
 if norm=='volscaled': f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_2_20270225_overnight_gap.csv',index=False)
