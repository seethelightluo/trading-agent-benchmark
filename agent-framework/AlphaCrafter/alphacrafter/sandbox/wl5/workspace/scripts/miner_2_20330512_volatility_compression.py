import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
series={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index(); p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v120=r.rolling(120,min_periods=80).std(); series[s]=pd.DataFrame({'f':-(v20/v120-1),'p':p})
all_dates=sorted(set().union(*[set(x.index) for x in series.values()])); results={}
for h in [5,10,20,40]:
  ic=[]; ns=[]; ds=[]
  for t in all_dates:
   va=[]; fw=[]
   for x in series.values():
    if t not in x.index: continue
    loc=x.index.get_loc(t)
    if isinstance(loc,slice) or loc+h>=len(x): continue
    f=x.iloc[loc].f; p0=x.iloc[loc].p; p1=x.iloc[loc+h].p
    if np.isfinite(f) and p0>0 and p1>0: va.append(f); fw.append(p1/p0-1)
   if len(va)>=8 and np.std(va)>0 and np.std(fw)>0: ic.append(np.corrcoef(va,fw)[0,1]);ns.append(len(va));ds.append(t)
  a=np.array(ic); results[h]=(a,ds,ns); print('horizon',h,'dates',len(a),'mean_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
a,ds,ns=results[10]
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-05-12')]:
 z=a[(np.array(ds)>=pd.Timestamp(lo))&(np.array(ds)<=pd.Timestamp(hi))]; print('regime',lo,hi,'n',len(z),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
