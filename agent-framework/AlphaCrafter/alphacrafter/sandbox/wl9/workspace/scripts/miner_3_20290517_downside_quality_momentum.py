import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); rows=[]
# 60d trend rewarded for positive breadth and penalized by downside volatility; 10d forecast.
for i,t in enumerate(p.index):
 if i<70 or i+10>=len(p): continue
 r60=p.iloc[i]/p.iloc[i-60]-1
 down=r.iloc[i-19:i+1].where(r.iloc[i-19:i+1]<0).std(ddof=1)*np.sqrt(20)
 breadth=(p.iloc[i]/p.iloc[i-5]-1>0).mean()
 sig=r60/down.replace(0,np.nan)
 # use continuous breadth as a mild common regime weight, not selection
 sig=sig*(0.5+breadth)
 f=p.iloc[i+10]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
A=pd.DataFrame(rows,columns=['date','ic','n']);x=A.ic.to_numpy();print('assets',len(D),'calendar_dates',len(p),'valid_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15);print('10d_ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for a,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]:
 z=A[(A.date>=l)&(A.date<h)].ic;print(a,'dates',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan,'hit',(z>0).mean() if len(z) else np.nan)
