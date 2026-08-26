import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
for h in [1,5,10,20,40]:
 a=[]
 for i,t in enumerate(p.index):
  if i<25 or i+h>=len(p): continue
  sig=-r.iloc[i-19:i+1].std(); f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8: a.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
 A=pd.DataFrame(a,columns=['date','ic']); x=A.ic
 print('H',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'online',round(A[A.date>=pd.Timestamp('2026-07-16')].ic.mean(),6))
