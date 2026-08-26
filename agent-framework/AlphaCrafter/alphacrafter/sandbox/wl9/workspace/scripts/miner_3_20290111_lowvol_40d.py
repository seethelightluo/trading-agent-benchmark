import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); out=[]
for i,t in enumerate(p.index):
 if i<50 or i+20>=len(p): continue
 sig=-r.iloc[i-39:i+1].std(); f=p.shift(-20).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8: out.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(out,columns=['date','n','ic']); print('assets',len(D),'dates',len(A),'mean_n',A.n.mean())
for label,c in [('full',A.date>=A.date.min()),('recent',A.date>=A.date.max()-pd.Timedelta(days=370)),('online',A.date>=pd.Timestamp('2026-07-16')),('2028',A.date>=pd.Timestamp('2028-01-01'))]:
 x=A[c].ic; print(label,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
