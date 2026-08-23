import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=2700)
 if x is None or len(x)<100: continue
 x=x.copy(); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); c=x.close.astype(float); r=c.pct_change()
 # Directional trend efficiency: signed cumulative return divided by realized path length.
 path=r.abs().rolling(20,min_periods=15).sum(); f=(r.rolling(20,min_periods=15).sum()/path).clip(-1,1)
 D[s]=pd.DataFrame({'f':f,'r1':c.pct_change().shift(-1),'r5':c.pct_change(5).shift(-5),'r10':c.pct_change(10).shift(-10)})
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 z=[(s,x.loc[d]) for s,x in D.items() if d in x.index and np.isfinite(x.loc[d,'f']) and np.isfinite(x.loc[d,'r1'])]
 if len(z)>=8: rows += [{'date':d,'s':s,**q.to_dict()} for s,q in z]
a=pd.DataFrame(rows); print('dates',a.date.nunique(),'rows',len(a),'avg_names',a.groupby('date').size().mean(),'coverage',len(a)/(a.date.nunique()*15))
for y in ['r1','r5','r10']:
 ic=a.groupby('date').apply(lambda g:g.f.corr(g[y])).dropna(); print(y,'obs',len(ic),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean())
for yr,g in a.groupby(a.date.dt.year):
 ic=g.groupby('date').apply(lambda q:q.f.corr(q.r1)).dropna(); print('regime',yr,'n',len(ic),'IC',round(ic.mean(),5),'ICIR',round(ic.mean()/ic.std(ddof=1),5))
