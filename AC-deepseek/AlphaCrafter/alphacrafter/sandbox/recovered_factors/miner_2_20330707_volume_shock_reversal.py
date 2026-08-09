import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; ds={}
for a in A:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv'); q.date=pd.to_datetime(q.date); q=q.set_index('date'); ds[a]=q
p=pd.DataFrame({a:ds[a].close for a in A}).sort_index(); vol=pd.DataFrame({a:ds[a].volume for a in A}).reindex(p.index)
r=p.pct_change(); rv=vol/vol.rolling(40,min_periods=20).median()-1
# abnormal volume with recent move fade; volume signal is lagged
sig=(-r.rolling(3,min_periods=3).sum()*np.log1p(rv.clip(lower=0))).shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; zc=[]; ns=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   zc.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(zc); print('H',h,'dates',len(x),'N',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for y in [2024,2025,2026,2027,2028,2029,2030,2031,2032,2033]:
 x=[]
 for dt in p.index[p.index.year==y]:
  z=pd.concat([sig.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if x: print('Y',y,len(x),round(np.mean(x),6),round(np.mean(x)/np.std(x,ddof=1),4))
