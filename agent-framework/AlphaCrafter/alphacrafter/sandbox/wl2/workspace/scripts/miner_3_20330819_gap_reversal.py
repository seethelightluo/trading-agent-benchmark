import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>=100:return d
  except Exception: pass
 return None
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}; print('loaded',len(xs),list(xs))
rows=[]; total=0
for s,d in xs.items():
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.sort_values('date'); total+=len(d)
 c=d.close.astype(float); o=d.open.astype(float); gap=np.log(o/c.shift(1)); vol=np.log(c/c.shift(1)).rolling(20).std()
 f=(-gap/vol).clip(lower=0,upper=4).shift(1); fr=c.shift(-1)/c-1
 rows.append(pd.DataFrame({'date':d.date,'s':s,'f':f,'r':fr}).dropna())
x=pd.concat(rows); ics=[]; counts=[]
for dt,g in x.groupby('date'):
 g=g.replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: ics.append(g.f.corr(g.r,method='spearman')); counts.append(len(g))
ic=pd.Series(ics).dropna(); ic.index=pd.to_datetime(sorted(x.date.unique())[-len(ic):]) if False else ic.index
print('dates',len(ic),'avg_n',np.mean(counts),'assets',x.s.nunique(),'coverage',len(x)/total)
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'median',ic.median())
# use grouped date indexed output for regimes
vals=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: vals.append((pd.Timestamp(dt),g.f.corr(g.r,method='spearman')))
ii=pd.Series(dict(vals)).dropna()
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-08-19')]:
 q=ii[(ii.index>=a)&(ii.index<=b)]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 rr=[]
 for s,d in xs.items():
  c=d.close.astype(float);o=d.open.astype(float);f=(-np.log(o/c.shift(1))/np.log(c/c.shift(1)).rolling(20).std()).clip(lower=0,upper=4).shift(1);rr.append(pd.DataFrame({'date':d.date,'f':f,'r':c.shift(-h)/c-1}).dropna())
 xx=pd.concat(rr); z=[]
 for dt,g in xx.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:z.append(g.f.corr(g.r,method='spearman'))
 z=pd.Series(z).dropna();print('h',h,'dates',len(z),'IC',z.mean(),'IR',z.mean()/z.std(ddof=1))
x.to_csv('scripts/miner_3_20330819_gap_reversal_signal.csv',index=False)
