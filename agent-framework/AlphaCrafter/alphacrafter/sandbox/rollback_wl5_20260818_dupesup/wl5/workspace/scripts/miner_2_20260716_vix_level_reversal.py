import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
pd0={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(pd0).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# Single interpretable idea: short-term reversal conditioned on absolute VIX regime.
# In stressed markets use reversal; in calm markets suppress it (state=VIX above trailing median).
state=(v>v.rolling(60,min_periods=40).median()).astype(float)
f=-r.rolling(5).sum().mul(state,axis=0)
for h in [1,5,10]:
 y=p.shift(-h)/p-1; a=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(a); print('h',h,'obs',len(a),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
# turnover and regime splits daily
rr=f.rank(axis=1,pct=True); print('turnover',rr.diff().abs().mean(axis=1).mean())
y=p.shift(-1)/p-1;a=[]; ds=[]
for d in p.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d)
a=np.array(a); ds=pd.DatetimeIndex(ds)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print('regime',lo,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('corr_rev5',f.stack().corr((-r.rolling(5).sum()).stack()))
