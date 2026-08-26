import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4000)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index() if d is not None and len(d) else pd.Series(dtype=float)
px=pd.DataFrame({s:g(s) for s in U}).sort_index(); r=px.pct_change(); vol=r.rolling(60,min_periods=30).std()*np.sqrt(60)
# medium-horizon reversal, normalized by long-run risk; add mild penalty to deep drawdowns
r20=px.pct_change(20); dd=(px/px.rolling(120,min_periods=60).max()-1).clip(-1,0)
sig=-r20/(vol+1e-6)*(1+0.25*dd.abs())
for h in [5,10,20]:
 a=[];ds=[];ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],(px.shift(-h)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if np.isfinite(q):a.append(q);ds.append(d);ns.append(len(z))
 a=np.array(a); ds=np.array(ds,dtype='datetime64[ns]')
 print('h',h,'dates',len(a),'mean_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:
  for lab,cut in [('recent252','2028-06-28'),('2029','2029-01-01')]:
   q=a[ds>=np.datetime64(cut)];print(lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
