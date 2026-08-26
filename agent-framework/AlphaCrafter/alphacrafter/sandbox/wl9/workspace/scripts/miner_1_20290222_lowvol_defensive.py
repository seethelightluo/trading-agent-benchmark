import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Defensive low-volatility signal: inverse 40-day realized volatility, cross-sectionally ranked
sig=-r.rolling(40).std(); rows=[]
for i,t in enumerate(p.index):
 if i<50 or i+10>=len(p): continue
 q=pd.concat([sig.iloc[i],p.shift(-10).iloc[i]/p.iloc[i]-1],axis=1).dropna(); q.columns=['s','f']
 if len(q)>=8 and q.s.nunique()>1 and q.f.nunique()>1: rows.append((t,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
for lab,cond in [('all',np.ones(len(A),bool)),('recent252',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',(A.date>=pd.Timestamp('2028-01-01'))&(A.date<pd.Timestamp('2029-01-01'))),('2029',A.date>=pd.Timestamp('2029-01-01'))]:
 x=A[cond].ic; print(lab,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 z=[]
 for i in range(50,len(p)-h):
  q=pd.concat([sig.iloc[i],p.shift(-h).iloc[i]/p.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,round(np.nanmean(z),6),len(z))
