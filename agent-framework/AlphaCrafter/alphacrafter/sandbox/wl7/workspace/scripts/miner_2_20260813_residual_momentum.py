import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d): raw[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.DataFrame(raw).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
F={}; R={}
for s in U:
 x=r[s]; n=60; cov=(x*m).rolling(n,min_periods=40).mean()-x.rolling(n,min_periods=40).mean()*m.rolling(n,min_periods=40).mean(); var=(m*m).rolling(n,min_periods=40).mean()-m.rolling(n,min_periods=40).mean()**2
 beta=cov/var.replace(0,np.nan); F[s]=(x-beta*m).rolling(20,min_periods=15).sum(); R[s]=p[s].shift(-1)/p[s]-1
F=pd.DataFrame(F); R=pd.DataFrame(R); ics=[]; h5=[]; h10=[]; nms=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1])); nms.append(len(z))
  for h,out in [(5,h5),(10,h10)]:
   q=pd.concat([F.loc[dt],(p.shift(-h)/p-1).loc[dt]],axis=1).dropna(); out.append(q.iloc[:,0].corr(q.iloc[:,1]) if len(q)>=8 else np.nan)
a=np.array(ics); rank=F.rank(pct=True); turn=rank.diff().abs().mean(axis=1).mean()
def st(x):
 x=np.array(x); x=x[np.isfinite(x)]; return len(x),np.mean(x),np.mean(x)/np.std(x,ddof=1)
print('dates',len(a),'avg_names',np.mean(nms),'coverage',np.mean(nms)/15,'IC',st(a),'hit',np.mean(a>0),'turnover',turn,'5d',st(h5),'10d',st(h10))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=[]
 for dt in F.index:
  if lo<=dt.year<=hi:
   z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('regime',lo,hi,st(q))
