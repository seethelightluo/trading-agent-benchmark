import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
v=get_index_daily_data('VIX',days=4000)
if v is None: raise RuntimeError('VIX unavailable')
v=v.assign(date=pd.to_datetime(v.date)).drop_duplicates('date').set_index('date').close.reindex(p.index).ffill()
# At date t, use only trailing data. Low-VIX trend versus high-VIX relative reversal.
rows=[]
for i,t in enumerate(p.index):
 if i<65 or i+10>=len(p): continue
 rv=v.iloc[max(0,i-60):i+1]; high=v.iloc[i]>rv.median()
 mom=p.iloc[i]/p.iloc[i-20]-1; vol=r.iloc[i-19:i+1].std().replace(0,np.nan)
 trend=mom/vol
 rel5=r.iloc[i-4:i+1].sum(); rev=-(rel5-rel5.median())
 sig=rev if high else trend
 f=p.shift(-10).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),int(high)))
A=pd.DataFrame(rows,columns=['date','n','ic','high']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4),'high_share',round(A.high.mean(),4))
for label,cond in [('full',A.date>=A.date.min()),('recent252',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',A.date>=pd.Timestamp('2028-01-01'))]:
 q=A[cond].ic; print(label,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]
 for i,t in enumerate(p.index):
  if i<65 or i+h>=len(p): continue
  high=v.iloc[i]>v.iloc[max(0,i-60):i+1].median(); mom=p.iloc[i]/p.iloc[i-20]-1; vol=r.iloc[i-19:i+1].std().replace(0,np.nan); trend=mom/vol; rel5=r.iloc[i-4:i+1].sum(); sig=(-(rel5-rel5.median())) if high else trend
  f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 z=pd.Series(vals); print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
