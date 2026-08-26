import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date)
  D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); rows=[]
# Relative strength: asset's 20d return minus contemporaneous equal-weight universe 20d return.
# This removes broad common beta and tests cross-asset leadership.
for i,t in enumerate(p.index):
 if i<25 or i+20>=len(p): continue
 ret20=p.iloc[i]/p.iloc[i-20]-1
 bench=ret20.mean()
 sig=ret20-bench
 f=p.shift(-20).iloc[i]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna(); q.columns=['s','f']
 if len(q)>=8 and q.s.nunique()>1: rows.append((t,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
for label,cond in [('full',A.date>=A.date.min()),('recent252',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',A.date>=pd.Timestamp('2028-01-01'))]:
 q=A[cond].ic; print(label,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
# decay using same signal and forward horizons
for h in [1,5,10,20]:
 rr=[]
 for i,t in enumerate(p.index):
  if i<25 or i+h>=len(p): continue
  ret20=p.iloc[i]/p.iloc[i-20]-1; sig=ret20-ret20.mean(); f=p.shift(-h).iloc[i]/p.iloc[i]-1
  q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rr.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 x=pd.Series(rr); print('H',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
