import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=2700)
 if x is None or len(x)<100: continue
 x=x.copy();x.date=pd.to_datetime(x.date);x=x.set_index('date').sort_index(); c=x.close.astype(float); h=x.high.astype(float);l=x.low.astype(float); prev=c.shift(1)
 tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1); atr=tr.rolling(20,min_periods=10).mean()
 f=np.sign(c.pct_change())*(tr/atr-1).clip(-3,3)
 D[s]=pd.DataFrame({'f':f,'r1':c.pct_change().shift(-1),'r5':c.pct_change(5).shift(-5)})
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 z=[(s,x.loc[d]) for s,x in D.items() if d in x.index and np.isfinite(x.loc[d,'f']) and np.isfinite(x.loc[d,'r1'])]
 if len(z)>=8:
  rows += [{'date':d,'s':s,**q.to_dict()} for s,q in z]
a=pd.DataFrame(rows);print('dates',a.date.nunique(),'rows',len(a),'avg_names',a.groupby('date').size().mean(),'coverage',len(a)/(a.date.nunique()*15))
for y in ['r1','r5']:
 ic=a.groupby('date').apply(lambda g:g.f.corr(g[y])).dropna();print(y,len(ic),ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean())
for yr,g in a.groupby(a.date.dt.year):
 ic=g.groupby('date').apply(lambda q:q.f.corr(q.r1)).dropna();print(yr,round(ic.mean(),4),round(ic.mean()/ic.std(ddof=1),4))
