import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; px={}
for a in A:
 try:
  d=get_stock_daily_data(a,days=2000)
  if d is not None and len(d)>80:px[a]=d.set_index('date').close.astype(float)
 except: pass
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change(); fac=-(p/p.shift(5)-1); ics=[]; ns=[]; prev=None; to=[]
for i in range(10,len(p)-1):
 z=pd.concat([fac.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.f.corr(z.y));ns.append(len(z)); q=fac.iloc[i].rank(pct=True)
  if prev is not None:to.append((q-prev).abs().mean())
  prev=q
x=np.array(ics);x=x[np.isfinite(x)]
print('dates',len(p),'instruments',len(px),'IC_obs',len(x),'meanIC',x.mean(),'std',x.std(ddof=1),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/len(A),'turnover',np.mean(to))
for h in (1,5,10):
 y=p.pct_change(h).shift(-h);q=[]
 for i in range(10,len(p)-h):
  z=pd.concat([fac.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(z.f.corr(z.y))
 q=np.array(q);q=q[np.isfinite(q)];print('decay',h,len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1))
