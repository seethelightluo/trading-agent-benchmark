import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4000); D[s]=x.assign(date=pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();A=[]
for i,t in enumerate(p.index):
 if i<35 or i+10>=len(p):continue
 x=r.iloc[i-4:i+1].sum(); base=r.iloc[i-19:i+1].std()*np.sqrt(5); sig=-(x-x.median())*(x.abs()>2*base)
 q=pd.concat([sig,p.shift(-10).iloc[i]/p.iloc[i]-1],axis=1).dropna()
 if len(q)>=8 and (q.iloc[:,0]!=0).sum()>=3:A.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(A,columns=['date','n','ic']);print('dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
for k,c in [('full',A.date>=A.date.min()),('recent',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',A.date>=pd.Timestamp('2028-01-01'))]:
 z=A[c].ic;print(k,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
