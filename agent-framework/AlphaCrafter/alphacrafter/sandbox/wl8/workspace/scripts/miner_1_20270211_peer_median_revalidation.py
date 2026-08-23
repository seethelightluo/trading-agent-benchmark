import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 d=get_stock_daily_data(s,days=2700)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=d.close.astype(float)
 frames[s]=pd.DataFrame({'r5':c.pct_change(5),'y1':c.pct_change().shift(-1),'y5':c.pct_change(5).shift(-5)})
all_dates=sorted(set().union(*[set(x.index) for x in frames.values()])); rows=[]
for dt in all_dates:
 vals=[]
 for s,x in frames.items():
  if dt in x.index and np.isfinite(x.loc[dt,'r5']): vals.append(x.loc[dt,'r5'])
 if len(vals)<8: continue
 for s,x in frames.items():
  if dt not in x.index: continue
  q=x.loc[dt]; peers=[v for t,v in [(t,frames[t].loc[dt,'r5']) for t in frames if dt in frames[t].index] if t!=s and np.isfinite(v)]
  if len(peers)>=7 and np.isfinite(q.y1): rows.append((dt,s,float(np.median(peers)),float(q.y1),float(q.y5)))
a=pd.DataFrame(rows,columns=['date','symbol','f','y1','y5']);print('dates',a.date.nunique(),'rows',len(a),'avg_names',a.groupby('date').size().mean(),'coverage',len(a)/(a.date.nunique()*15))
for col in ['y1','y5']:
 ic=a.groupby('date').apply(lambda g:g.f.corr(g[col])).dropna(); print(col,'obs',len(ic),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
print('turnover',a.sort_values(['symbol','date']).groupby('date').f.rank(pct=True).groupby(a.date).apply(lambda x: np.nan))
for yr,g in a.groupby(a.date.dt.year):
 ic=g.groupby('date').apply(lambda q:q.f.corr(q.y1)).dropna(); print('regime',yr,'dates',len(ic),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1))
