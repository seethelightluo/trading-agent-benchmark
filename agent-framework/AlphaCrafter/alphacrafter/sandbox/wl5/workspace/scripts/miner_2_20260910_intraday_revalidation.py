import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d[d.date<=END].drop_duplicates('date').sort_values('date').set_index('date')
D={s:load(s) for s in U}; dates=sorted(set.intersection(*[set(x.index) for x in D.values()]));
o=pd.DataFrame({s:D[s].reindex(dates).open for s in U}); c=pd.DataFrame({s:D[s].reindex(dates).close for s in U});
f=-(c/o-1).ewm(span=3,min_periods=3,adjust=False).mean(); y=c.pct_change().shift(-1)
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15'),('2026-01-01','2026-07-15')]:
 a=[]; ns=[]
 for dt in f.loc[lo:hi].index:
  z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.asarray(a);print(lo,hi,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
