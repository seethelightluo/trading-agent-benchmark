import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={};vol={}
for a in A:
 d=get_stock_daily_data(a,days=4000)
 if d is not None:
  z=d.set_index('date');px[a]=z.close.astype(float);vol[a]=z.volume.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); vv=pd.concat(vol,axis=1).reindex(p.index).ffill();r=p.pct_change()
# volume surprise: return reversal weighted by abnormal volume, testing whether capitulation/reaction is predictive
vs=vv/vv.rolling(20,min_periods=10).median()-1
f=-p.pct_change(3)*vs.clip(-2,2)
for h in [1,5,10]:
 obs=[]; ns=[]; tr=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   obs.append(q.f.corr(q.y));ns.append(len(q))
   if i:
    z=pd.concat([f.iloc[i],f.iloc[i-1]],axis=1).dropna()
    if len(z)>=8:tr.append((z.iloc[:,0].rank()!=z.iloc[:,1].rank()).mean())
 x=np.array(obs);print('h',h,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',np.mean(ns)/15,'turn',np.mean(tr))
print('period',p.index.min().date(),p.index.max().date())
