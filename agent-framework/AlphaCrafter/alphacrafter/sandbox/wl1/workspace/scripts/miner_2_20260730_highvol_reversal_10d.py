import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is None or len(d)<100:d=get_index_daily_data(s,2500)
 if d is not None:F[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date)).sort_index()
p=pd.DataFrame(F).sort_index().ffill(); r=p.pct_change(); rv=r.rolling(20).std(); short=p/p.shift(5)-1
# Contrarian shock, activated by own volatility relative to cross-sectional median
vrel=rv.div(rv.median(axis=1),axis=0)
fac=-short*np.log1p(vrel.clip(lower=0))
for h in [1,5,10]:
 f=p.shift(-h)/p-1; vals=[]; ds=[]
 for t in fac.index:
  z=pd.concat([fac.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(t)
 a=np.array(vals);a=a[np.isfinite(a)];print('horizon',h,'dates',len(a),'ic',a.mean(),'std',a.std(ddof=1),'ir',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for y in sorted(set(pd.to_datetime(ds).year)):
 a=np.array([v for v,t in zip(vals,ds) if pd.Timestamp(t).year==y]);a=a[np.isfinite(a)];print('year',y,'n',len(a),'ic',a.mean(),'ir',a.mean()/a.std(ddof=1))
print('coverage',fac.notna().sum(axis=1).mean()/15)
