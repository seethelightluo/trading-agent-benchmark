import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=2200)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float).pct_change()
R=pd.DataFrame(D).sort_index(); m=R.mean(axis=1); out=[]
for i in range(61,len(R)-10):
 hist=R.iloc[i-60:i]; mm=m.iloc[i-60:i]; mask=mm<0
 if mask.sum()<10: continue
 vals={}
 for s in U:
  x=hist[s]; ok=mask & x.notna() & mm.notna()
  if ok.sum()<10 or np.var(mm[ok])==0: continue
  beta=np.cov(x[ok],mm[ok],ddof=1)[0,1]/np.var(mm[ok],ddof=1); vals[s]=-beta
 f=pd.Series(vals)
 for h in [1,5,10]:
  y=R.iloc[i:i+h].sum(axis=0); q=pd.concat([f,y.rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((R.index[i],h,len(q),q.iloc[:,0].corr(q.y,method='spearman')))
A=pd.DataFrame(out,columns=['date','h','n','ic']); print('dates',len(R),'instruments',len(D),'observations',len(A))
for h,g in A.groupby('h'): print(h,'ic_dates',len(g),'mean_n',round(g.n.mean(),2),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4),'hit',round((g.ic>0).mean(),4))
online=A[A.date>=pd.Timestamp('2026-07-16')]
for h,g in online.groupby('h'): print('online',h,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4),'hit',round((g.ic>0).mean(),4))
