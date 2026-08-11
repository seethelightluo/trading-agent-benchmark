import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2400)
 if d is None or len(d)<200:d=get_index_daily_data(s,2400)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows);w=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=w.pct_change()
# Conditional medium reversal: fade the 15-day move, normalized by 90-day risk,
# but suppress weak moves that are not unusually large cross-sectionally.
rv=r.rolling(90,min_periods=60).std()
z=w.pct_change(15)/(rv*np.sqrt(15)+1e-8)
cs=z.sub(z.median(axis=1),axis=0)
f=(-z.where(cs.abs().ge(0.5),0.0)).clip(-8,8)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1; qs=[];ns=[];idx=w.index
 if a:idx=idx[(idx>=a)&(idx<=b)]
 for dt in idx:
  x=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   q=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(x))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-04-07')]:print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_fraction', (f!=0).mean().mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270408_conditional_medium_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
