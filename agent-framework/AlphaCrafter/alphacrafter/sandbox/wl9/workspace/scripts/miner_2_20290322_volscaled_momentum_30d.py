import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date)
  D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
rows=[]
# Medium-horizon trend strength: 30d return divided by 20d realized volatility.
# Signal is known at t; forward returns begin at t+1 and end at t+h.
for i,t in enumerate(p.index):
 if i<35 or i+20>=len(p): continue
 ret=p.iloc[i]/p.iloc[i-30]-1
 vol=r.iloc[i-19:i+1].std()*np.sqrt(20)
 sig=ret/vol.replace(0,np.nan)
 for h in [5,10,20]:
  f=p.iloc[i+h]/p.iloc[i]-1
  q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   rows.append((t,h,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'dates',len(p))
for h in [5,10,20]:
 q=A[A.h==h]; x=q.ic.to_numpy(); print('H',h,'valid_dates',len(q),'mean_n',q.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
 for label,lo,hi in [('2020-23','2020','2024'),('2024-26','2024','2027'),('2027-29','2027','2030'),('2028-29','2028','2030')]:
  y=q[(q.date>=lo)&(q.date<hi)].ic.to_numpy()
  print(' ',label,len(y),np.mean(y) if len(y) else np.nan, np.mean(y)/np.std(y,ddof=1)*np.sqrt(252) if len(y)>1 else np.nan)
