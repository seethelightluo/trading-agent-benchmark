import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
ds={s:load(s) for s in U}; close=pd.concat({s:d.close for s,d in ds.items()},axis=1).sort_index(); vol=pd.concat({s:d.volume for s,d in ds.items()},axis=1).reindex(close.index); ret=close.pct_change()
# Volume shock reversal: unusually high recent volume, signed by recent return; fade price moves on volume shocks
for w in [5,10,20]:
 z=(vol/vol.rolling(60,min_periods=20).median()-1)*(-ret.rolling(w).sum())
 for h in [1,5,10]:
  ic=[]; ns=[]; dates=[]
  for i in range(60,len(close)-h):
   q=pd.concat([z.iloc[i],(close.iloc[i+h]/close.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));dates.append(close.index[i])
  a=np.array(ic); print('w',w,'h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
  if h==10: print('regime',{int(y):round(a[np.array([d.year for d in dates])==y].mean(),5) for y in sorted(set(d.year for d in dates))})
