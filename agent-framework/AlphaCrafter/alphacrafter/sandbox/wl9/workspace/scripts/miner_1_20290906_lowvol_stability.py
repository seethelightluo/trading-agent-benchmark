import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3600)
 if x is not None and len(x): D[s]=x.sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Candidate: stable low volatility, rewarded by next 10d returns
fac=-r.rolling(20).std()
def run(f,h):
 out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(p.shift(-h).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1]))
 x=pd.Series(out); return x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x)),(x>0).mean(),len(x)
q=fac.dropna(how='all'); print('dates',len(q),'assets',len(D),'coverage',q.notna().sum(axis=1).mean()/15,'turnover',q.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10,20]: print('h',h,run(fac,h))
for name,a,b in [('full',None,None),('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029','2029-01-01',None),('recent252',None,'recent')]:
 f=fac
 if name=='recent252':
  # evaluate only last 252 factor dates
  f=fac.tail(252)
 else: f=fac.loc[slice(a,b)]
 print(name,run(f,10))
