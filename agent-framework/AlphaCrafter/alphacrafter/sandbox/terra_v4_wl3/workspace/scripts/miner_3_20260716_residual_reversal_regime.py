import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
p=pd.DataFrame(D).sort_index();r=p.pct_change(); x=r.rolling(20,min_periods=15).sum(); fac=-(x-x.median(axis=1))
fwd=r.shift(-1)
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 vals=[]
 for dt in fac.index:
  if pd.Timestamp(lo)<=dt<=pd.Timestamp(hi):
   a=fac.loc[dt];b=fwd.loc[dt];ok=a.notna()&b.notna()
   if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic)
 print(lo, 'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1),'N',len(vals))
