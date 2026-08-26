import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; EQ=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']; D={}
for s in U:
 x=get_stock_daily_data(s,days=2200); z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.pct_change()
R=pd.DataFrame(D).sort_index(); E=R[EQ].mean(axis=1); out=[]
for i in range(25,len(R)-10):
 f=R.iloc[i-20:i].sum()-E.iloc[i-20:i].sum(); ybase=R.iloc[i:i+10].sum(axis=0)
 for h in [1,5,10]:
  y=R.iloc[i:i+h].sum(); q=pd.concat([f,y.rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((R.index[i],h,len(q),q.iloc[:,0].corr(q.y,method='spearman')))
A=pd.DataFrame(out,columns=['date','h','n','ic']); print('dates',len(R),'instruments',len(D),'observations',len(A))
for h,g in A.groupby('h'): print(h,len(g),round(g.n.mean(),2),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),4),round((g.ic>0).mean(),4))
print('online')
for h,g in A[A.date>=pd.Timestamp('2026-07-16')].groupby('h'): print(h,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),4))
