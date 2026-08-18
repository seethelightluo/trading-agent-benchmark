import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill();r=p.pct_change();v=r.rolling(20).std(); cs=v.mean(1); med=cs.rolling(252).median()
for name,mask in [('low',cs<=med),('high',cs>med),('all',pd.Series(True,index=p.index))]:
 f=-p.pct_change(3); f=f.where(mask,0.0); x=[]; cov=[]
 for i in range(len(p)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8:x.append(q.f.corr(q.y));cov.append(len(q)/15)
 x=np.array(x); print(name,len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0),np.mean(cov))
