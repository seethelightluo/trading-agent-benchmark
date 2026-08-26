import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; c={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is not None and len(d): c[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date))
p=pd.DataFrame(c).sort_index(); r=p.pct_change();
# medium horizon risk-adjusted trend: lagged 60d excess return divided by realized vol, neutralized cross-section
f=r.rolling(60,min_periods=40).sum()/r.rolling(60,min_periods=40).std()
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(a,index=ds).dropna();print(h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q),'N',np.mean(ns))
print('coverage',f.notna().mean().mean(),'last',p.index.max())
# regime
h=10;y=p.shift(-h)/p-1;a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.Series(dict(a))
for lo,hi in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 x=q[(q.index>=lo)&(q.index<=hi)]; print('REG',lo,hi,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20341127_riskadj60_signal.csv',index=False)
