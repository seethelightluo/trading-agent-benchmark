import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}; r=pd.DataFrame(p).pct_change()
# Breadth-confirmed medium-term momentum: favor 20d trend when market breadth is strong,
# reverse it when breadth is weak. Breadth and trend lagged one day.
breadth=(r.rolling(20).sum()>0).mean(axis=1); market=r.mean(axis=1).rolling(20).sum();
sig=r.rolling(20).sum().where(breadth.shift(1)>=.5,-r.rolling(20).sum()).shift(1); f=r.shift(-1)
for h in [1,3,5,10]:
 y=r.shift(-1).rolling(h).sum().shift(-(h-1)); ic=[]; cov=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);cov.append(len(z)/15)
 q=np.array(ic);print('h',h,'dates',len(q),'coverage',np.mean(cov),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
# regime halves and breadth buckets
for lo,hi in [(0,.34),(.34,.67),(.67,1.01)]:
 ic=[]
 for d in sig.index:
  if not(lo<=breadth.shift(1).get(d,np.nan)<hi):continue
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(ic);print('breadth',lo,hi,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('turnover',np.nanmean(np.abs(sig.rank(axis=1,pct=True).diff()).sum(axis=1)/2))
sig.to_csv('scripts/miner_3_20330318_breadth_confirmed_signal.csv')
