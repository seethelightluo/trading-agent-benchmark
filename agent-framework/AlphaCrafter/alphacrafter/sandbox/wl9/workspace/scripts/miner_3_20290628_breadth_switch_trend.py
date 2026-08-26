import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();rows=[]
def sig(i):
 if i<65:return None
 b=(r.iloc[i-4:i+1]>0).mean().mean();ret=p.iloc[i]/p.iloc[i-20]-1;vol=r.iloc[i-19:i+1].std().replace(0,np.nan)
 return (1 if b>=.5 else -1)*ret/vol
for i,t in enumerate(p.index):
 if i<65 or i+10>=len(p):continue
 s=sig(i);f=p.shift(-10).iloc[i]/p.iloc[i]-1;q=pd.concat([s.rename('signal'),f.rename('fwd')],axis=1).dropna()
 if len(q)>=8 and q.signal.nunique()>1:rows.append((t,len(q),q.signal.rank().corr(q.fwd.rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']);print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
for name,c in [('full',A.date>=A.date.min()),('online',A.date>=pd.Timestamp('2026-07-16')),('recent',A.date>=A.date.max()-pd.Timedelta(days=370)),('2027+',A.date>=pd.Timestamp('2027-01-01'))]:
 q=A[c].ic;print(name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 z=[]
 for i in range(65,len(p)-h):
  s=sig(i);f=p.shift(-h).iloc[i]/p.iloc[i]-1;q=pd.concat([s.rename('signal'),f.rename('fwd')],axis=1).dropna()
  if len(q)>=8 and q.signal.nunique()>1:z.append(q.signal.rank().corr(q.fwd.rank()))
 z=pd.Series(z);print('decay',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
