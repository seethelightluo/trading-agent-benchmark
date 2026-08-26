import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index();P[s]=d.close;V[s]=d.volume
px=pd.DataFrame(P).sort_index().loc[:'2028-04-05']; vol=pd.DataFrame(V).reindex(px.index)
r=px.pct_change(); vr=vol/vol.rolling(20,min_periods=10).mean()
fac=-px.pct_change(3)*(1+np.log(vr.clip(lower=.25,upper=4)))
for h in [1,5,10]:
 f=px.shift(-h)/px-1;a=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=np.asarray(a);ds=pd.DatetimeIndex(ds)
 def m(x):return len(x),float(x.mean()),float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))),float((x>0).mean())
 print('horizon',h,'all',m(a),'online',m(a[ds>='2026-07-16']),'recent',m(a[ds>='2027-04-01']),'mean_n',round(float(np.mean(ns)),2))
print('coverage',round(float(fac.notna().mean().mean()),4),'turnover',round(float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),5))