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
for i,t in enumerate(p.index):
 if i<80 or i+10>=len(p): continue
 c=p.iloc[i]; vol=r.iloc[i-19:i+1].std().replace(0,np.nan)
 mom=(c/p.iloc[i-10]-1)/vol
 # Trend continuation is stronger when the cross-asset median 10d return is positive;
 # attenuate signal in the opposite breadth regime.
 breadth=(c/p.iloc[i-10]-1).median()
 sig=(mom if breadth>0 else mom*0.35).replace([np.inf,-np.inf],np.nan)
 for h in [1,5,10]:
  f=p.iloc[i+h]/c-1; q=pd.concat([sig,f],axis=1).dropna(); q.columns=['s','f']
  if len(q)>=8: rows.append((t,h,len(q),q.s.rank().corr(q.f.rank()),int(breadth>0)))
A=pd.DataFrame(rows,columns=['date','h','n','ic','gate']); print('period',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(A))
for h in [1,5,10]:
 q=A[A.h==h]; x=q.ic.dropna(); print('H',h,'dates',len(x),'mean_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'gate_frac',round(q.gate.mean(),4))
 for label,cond in [('recent252',q.date>=q.date.max()-pd.Timedelta(days=370)),('online',q.date>=pd.Timestamp('2026-07-16'))]:
  y=q.loc[cond,'ic'].dropna(); print(label,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6))
