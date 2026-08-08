import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill(); sh=v.pct_change(5)
for q in [.65,.75,.85]:
 for lb in [3,10]:
  gate=sh>sh.rolling(60,min_periods=30).quantile(q); f=-p.pct_change(lb).where(gate,np.nan); y=p.shift(-1)/p-1; a=[];ns=[]
  for d in f.index:
   z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>=3 and z.iloc[:,1].nunique()>=3:
    x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(x):a.append(x);ns.append(len(z))
  a=np.array(a);print('q',q,'lb',lb,'dates',len(a),'IC %.6f ICIR %.6f hit %.4f n %.2f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns)))
