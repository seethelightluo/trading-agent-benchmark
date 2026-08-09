import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=2470)
 if d is not None: P[s]=d.set_index('date').close.astype(float)
P=pd.concat(P,axis=1).sort_index().ffill(); R=P.pct_change()
v=get_index_daily_data('VIX',days=2470).set_index('date').close.astype(float).reindex(P.index).ffill()
shock=v.pct_change(5).clip(-.5,.5)
f=(-P.pct_change(5).mul(1+shock,axis=0)).where(shock>0,-P.pct_change(5)*.5)
Y=P.shift(-1).div(P)-1; obs=[]; ns=[]
for dt in P.index:
 q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1: obs.append(q.f.corr(q.y));ns.append(len(q))
a=np.array(obs); a=a[np.isfinite(a)]
print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for yr in range(2020,2027):
 z=[]
 for dt in P.index[P.index.year==yr]:
  q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(q.f.corr(q.y))
 z=np.array(z);print('year',yr,'n',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1))
# factor rank correlation with plain 5d reversal and existing momentum proxy
print('corr_reversal',pd.concat([f.stack(),(-P.pct_change(5)).stack()],axis=1).dropna().corr().iloc[0,1])
