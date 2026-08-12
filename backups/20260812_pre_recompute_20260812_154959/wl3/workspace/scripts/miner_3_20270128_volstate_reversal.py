import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<200:d=get_index_daily_data(s,2300)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows);w=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=w.pct_change()
# Volatility-state reversal: short-term reversal receives a smooth boost when recent risk is elevated.
# This targets transient overreaction while retaining a simple, interpretable price-only signal.
v=r.rolling(20,min_periods=12).std(); cross=v.rank(axis=1,pct=True)
f=(-r)*(0.65+0.70*cross).clip(-3,3)
def calc(h,st=None,en=None):
 fut=w.shift(-h)/w-1; qs=[];ns=[];idx=w.index
 if st:idx=idx[(idx>=st)&(idx<=en)]
 for dt in idx:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]:print('H',h,'n mean std ICIR hit avgN',calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27')]:print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum(),'rank_turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270128_volstate_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
