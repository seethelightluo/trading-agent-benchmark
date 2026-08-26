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
# Volatility-scaled medium trend: 60d excess return divided by recent 20d realized volatility.
sig=p.pct_change(60)/(r.rolling(20).std()*np.sqrt(20)+1e-12)
for i,t in enumerate(p.index):
 if i<80 or i+20>=len(p): continue
 q=pd.concat([sig.iloc[i],p.shift(-20).iloc[i]/p.iloc[i]-1],axis=1).dropna(); q.columns=['s','f']
 if len(q)>=8 and q.s.nunique()>1: rows.append((t,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(A),'mean_n',round(A.n.mean(),2),'coverage',round(A.n.mean()/15,4))
x=A.ic
print('IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for label,cond in [('recent252',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',A.date>=pd.Timestamp('2028-01-01'))]:
 y=A[cond].ic; print(label,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
