import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is None:return None
 return d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:L(s) for s in U if L(s) is not None},axis=1).sort_index(); r=np.log(P/P.shift(1))
f=((r.rolling(20).std()*np.sqrt(20))+(r.rolling(60).std()*np.sqrt(60)))/2; f=f.shift(1)
for h in [5,10,20,40]:
 fw=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);a=a[np.isfinite(a)]
 print(h,len(a),round(np.mean(ns),2),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),3),round(np.mean(a>0),4))
out=[]
for dt in f.index:
 for s in f:
  if pd.notna(f.loc[dt,s]):out.append([str(dt.date()),s,float(f.loc[dt,s])])
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20341221_blended_volatility_premium_signal.csv',index=False)
