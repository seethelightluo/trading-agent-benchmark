import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
G={'000300.SH':'eq','SPX':'eq','HSI':'eq','N225':'eq','SX5E':'eq','000688.SH':'eq','SOX':'eq','NDX':'eq','XAU':'com','COPPER':'com','WTI':'com','BTC':'cry','ETH':'cry','US10Y':'rate','CN10Y':'rate'}
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows);w=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=w.pct_change()
grp=pd.DataFrame({g:r.filter(items=[s for s in U if G[s]==g]).median(axis=1) for g in set(G.values())})
rel=pd.DataFrame(index=r.index,columns=U,dtype=float)
for s in U: rel[s]=r[s]-grp[G[s]]
vol=r.rolling(30,min_periods=20).std()
# Medium-horizon cluster-relative reversal, volatility normalized, with mild recent confirmation
f=-(rel.rolling(8,min_periods=8).sum()/vol.clip(lower=1e-5))
confirm=np.sign(rel.rolling(3,min_periods=3).sum())
f=f*(1+0.25*confirm*np.sign(-f))
f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1; qs=[];ns=[];idx=w.index
 if a:idx=idx[(idx>=a)&(idx<=b)]
 for dt in idx:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(w.index.max().date()))]:print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20271202_medium_cluster_reversal_signal.csv',index=False)
