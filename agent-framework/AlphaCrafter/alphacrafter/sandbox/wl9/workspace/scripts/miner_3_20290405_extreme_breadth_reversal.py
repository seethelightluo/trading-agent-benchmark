import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); rows=[]
for i,t in enumerate(p.index):
 if i<70 or i+5>=len(p):continue
 rr=p.iloc[i]/p.iloc[i-5]-1;b=(rr>0).mean(); h=[]
 for k in range(i-60,i):
  q=p.iloc[k]/p.iloc[k-5]-1;h.append((q>0).mean())
 if b>np.nanpercentile(h,20):continue
 sig=-(rr-rr.median());f=p.iloc[i+5]/p.iloc[i]-1;q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','ic']);x=A.ic.to_numpy();print('assets',len(D),'dates',len(p),'gated',len(A));print('ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for a,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]:
 z=A[(A.date>=l)&(A.date<h)].ic;print(a,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
