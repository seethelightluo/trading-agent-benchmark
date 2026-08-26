import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); rows=[]
# Stress-conditioned reversal normalized by each asset's recent realized volatility.
for i,t in enumerate(p.index):
 if i<70 or i+5>=len(p):continue
 r=p.iloc[i]/p.iloc[i-5]-1; breadth=(r>0).mean(); hist=[(p.iloc[k]/p.iloc[k-5]-1>0).mean() for k in range(i-60,i)]
 if breadth>np.nanpercentile(hist,20):continue
 vol=p.iloc[i-10:i].pct_change().std().replace(0,np.nan)
 sig=-(r-r.median())/vol; f=p.iloc[i+5]/p.iloc[i]-1;q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
A=pd.DataFrame(rows,columns=['date','ic','n']);x=A.ic.to_numpy();print('assets',len(D),'dates',len(p),'valid_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15);print('5d_ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for a,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]:
 z=A[(A.date>=l)&(A.date<h)].ic;print(a,'dates',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan,'hit',(z>0).mean() if len(z) else np.nan)
