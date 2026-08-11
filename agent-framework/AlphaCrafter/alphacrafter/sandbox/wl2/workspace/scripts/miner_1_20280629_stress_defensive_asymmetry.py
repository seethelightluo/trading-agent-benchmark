import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); P[s]=d.set_index('date')['close'].sort_index()
px=pd.concat(P,axis=1).sort_index().ffill(); r=np.log(px).diff(); cs=r.mean(axis=1)
# 20d downside asymmetry, activated by a lagged stress percentile; avoid cross-asset mean NaNs
mv=cs.rolling(20,min_periods=15).std(); base=mv.rolling(120,min_periods=60).median(); stress=(mv/base).replace([np.inf,-np.inf],np.nan)
vol=r.rolling(20,min_periods=15).std(); dn=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
f=((1-dn/vol)*stress).shift(1)
for h in [1,3,5,10]:
 y=np.log(px).shift(-h)-np.log(px); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 a=np.asarray(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('instruments',len(P),'coverage',round(f.notna().sum(axis=1).mean()/len(P),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
