import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();p=pd.to_numeric(d.close,errors='coerce'); vol=pd.to_numeric(d.volume,errors='coerce'); r=p.pct_change(); vz=(vol-vol.rolling(60,min_periods=30).mean())/vol.rolling(60,min_periods=30).std(); C[s]=pd.DataFrame({'f':-(r.rolling(10,min_periods=8).sum())*vz,'p':p})
D=sorted(set().union(*[set(x.index) for x in C.values()])); out=[]
for t in D:
 v=[];y=[]
 for x in C.values():
  if t not in x.index:continue
  j=x.index.get_loc(t)
  if isinstance(j,slice) or j+10>=len(x):continue
  f=x.iloc[j].f;p=x.iloc[j].p;p2=x.iloc[j+10].p
  if np.isfinite(f) and p>0 and p2>0:v.append(f);y.append(p2/p-1)
 if len(v)>=8 and np.std(v)>0 and np.std(y)>0:out.append((pd.Timestamp(t),np.corrcoef(v,y)[0,1],len(v)))
a=np.array([z[1] for z in out]); ds=pd.to_datetime([z[0] for z in out]); ns=np.array([z[2] for z in out]);print('dates',len(a),'mean_instruments',ns.mean(),'coverage',ns.mean()/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-05-12')]:
 z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print(lo,hi,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
