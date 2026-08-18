import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill();r=p.pct_change()
for look in [1,2,3,4,7,10]:
 f=-r.rolling(look).sum(); xs=[];ns=[];turn=[];prev=None
 for i in range(len(p)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y));ns.append(len(q))
  z=f.iloc[i].rank(pct=True)
  if prev is not None:turn.append(np.nanmean(abs(z-prev)))
  prev=z
 x=np.array(xs);x=x[np.isfinite(x)];print(look,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'cov',np.mean(ns)/15,'turn',np.nanmean(turn))
