import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4100)
 if d is None or len(d)<200: d=get_index_daily_data(s,days=4100)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
med=r.rolling(5).median().mean(axis=1)
disp=r.rolling(5).std().mean(axis=1)
stress=(med<0)&(disp>disp.rolling(120).median())
base=-r.rolling(5).sum()
f=base.sub(base.median(axis=1),axis=0).mul(stress.astype(float),axis=0).shift(1)
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
print('active',round(stress.mean(),4),'coverage',round(f.notna().mean().mean(),4))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_1_20350622_stress_dispersion_reversal_signal.csv',index=False)
