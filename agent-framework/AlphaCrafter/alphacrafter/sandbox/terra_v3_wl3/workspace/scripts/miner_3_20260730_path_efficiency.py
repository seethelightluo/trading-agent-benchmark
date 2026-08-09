import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].sort_index() for s in U}
# Candidate: downside-normalized path efficiency: net return divided by absolute return path.
F={s:P[s].pct_change(1).rolling(20,min_periods=15).sum()/(P[s].pct_change(1).abs().rolling(20,min_periods=15).sum()+1e-12) for s in U}
f=pd.concat(F,axis=1).sort_index(); vals={h:[] for h in [1,5,10]}; dates={h:[] for h in [1,5,10]}; ns={h:[] for h in [1,5,10]}
for dt in f.index:
 for h in vals:
  xs=[];ys=[]
  for s in U:
   j=P[s].index.searchsorted(dt)
   if j<len(P[s]) and P[s].index[j]==dt and j+h<len(P[s]) and pd.notna(F[s].loc[dt]): xs.append(F[s].loc[dt]); ys.append(P[s].iloc[j+h]/P[s].iloc[j]-1)
  if len(xs)>=8:
   vals[h].append(spearmanr(xs,ys).statistic); dates[h].append(dt); ns[h].append(len(xs))
for h in vals:
 a=pd.Series(vals[h],index=dates[h]); print('H',h,'dates',len(a),'avgN',np.mean(ns[h]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',sum(ns[h])/(len(a)*15)); print('regimes',a.groupby(a.index.year).mean().round(4).to_dict())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean()*2)
for w,n in [(5,'rev'),(20,'mom')]: print('rho',n,f.stack().corr(pd.concat({s:P[s].pct_change(w) for s in U},axis=1).stack()))
