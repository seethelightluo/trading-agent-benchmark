import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date)
  D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); rows=[]
# Stress-conditioned residual reversal: remove the contemporaneous cross-sectional
# component from each 5d move, then buy unusually weak assets during broad stress.
for i,t in enumerate(p.index):
 if i<80 or i+10>=len(p): continue
 r5=p.iloc[i]/p.iloc[i-5]-1
 breadth=(r5>0).mean()
 hist=np.array([(p.iloc[k]/p.iloc[k-5]-1>0).mean() for k in range(i-60,i)])
 # stress when today's breadth is below its trailing 20th percentile
 if breadth>np.nanpercentile(hist,20): continue
 resid=r5-r5.median()
 rr=p.iloc[i-20:i].pct_change()
 scale=rr.std().replace(0,np.nan)
 sig=-resid/scale
 f=p.iloc[i+10]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
A=pd.DataFrame(rows,columns=['date','ic','n']); x=A.ic.to_numpy()
def met(z):
 z=np.asarray(z); return (len(z),float(np.nanmean(z)),float(np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float(np.mean(z>0)))
print('assets',len(D),'calendar_dates',len(p),'valid_dates',len(A),'mean_n',A.n.mean(),'coverage',A.n.mean()/15)
print('10d',met(x))
for name,l,h in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]: print(name,met(A[(A.date>=l)&(A.date<h)].ic))
for h in [5,20,40]:
 z=[]
 for i,t in enumerate(p.index):
  if i<80 or i+h>=len(p): continue
  r5=p.iloc[i]/p.iloc[i-5]-1; breadth=(r5>0).mean(); hist=np.array([(p.iloc[k]/p.iloc[k-5]-1>0).mean() for k in range(i-60,i)])
  if breadth>np.nanpercentile(hist,20): continue
  sig=-(r5-r5.median())/p.iloc[i-20:i].pct_change().std().replace(0,np.nan)
  f=p.iloc[i+h]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna()
  if len(q)>=8: z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,met(z))
