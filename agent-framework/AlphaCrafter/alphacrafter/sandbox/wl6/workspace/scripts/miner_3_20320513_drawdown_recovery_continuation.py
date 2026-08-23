import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-05-12')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:cut]
r=P.pct_change(); ret20=P/P.shift(20)-1
# Continuation favors positive trend, while recovery position penalizes assets still near a deep 120d drawdown.
peak=P.rolling(120,min_periods=60).max(); dd=P/peak-1
recovery=(P/P.shift(20)-1) - 0.35*(-dd).clip(lower=0)
vol=r.rolling(40,min_periods=20).std()*np.sqrt(40)
f=recovery/(vol+1e-12)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'coverage',f.notna().stack().mean())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): ics.append(q); ns.append(len(z)); ds.append(dt)
 a=np.asarray(ics); print('h',h,'valid_dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 q=pd.Series(a,index=ds); print('regimes',q.groupby(q.index.year).mean().round(6).to_dict())
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean())
