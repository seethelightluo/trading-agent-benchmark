import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change(); m=r.mean(axis=1)
raw=px.pct_change(20).shift(1); mr=m.rolling(20).sum().shift(1)
sig=raw.sub(mr,axis=0); vol=r.rolling(20).std().shift(1)*np.sqrt(20); f=sig/vol
for h in [1,3,5,10]:
 fr=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); print(h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'assets',len(D),'dates',len(px))
f.to_csv('scripts/miner_1_20330902_residual_trend_signal.csv')

if __name__ == '__main__': pass
