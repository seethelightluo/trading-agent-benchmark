import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
ds={s:load(s) for s in U}; close=pd.concat({s:d.close for s,d in ds.items()},axis=1).sort_index(); z=pd.DataFrame(index=close.index)
for s,d in ds.items():
 r=d.close.pct_change(); med=d.volume.rolling(60,min_periods=20).median(); z[s]=(d.volume/med-1)*(-r.rolling(5,min_periods=5).sum())
print('dates',len(close),'instruments',len(U))
for h in [1,5,10]:
 ic=[];ns=[];dates=[]
 for dt in close.index:
  if dt not in z.index: continue
  j=close.index.get_loc(dt); fut=close.iloc[j+h]/close.iloc[j]-1 if j+h<len(close) else pd.Series(dtype=float)
  q=pd.concat([z.loc[dt],fut.rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));dates.append(dt)
 a=np.asarray(ic);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('regime',{y:round(a[np.array([d.year for d in dates])==y].mean(),5) for y in sorted(set(d.year for d in dates))})
