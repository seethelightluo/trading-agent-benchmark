import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=cl.pct_change()
# Relative momentum: asset 40-session return minus contemporaneous cross-sectional median return.
# The one-day lag ensures only completed information is used.
raw=cl.pct_change(40)
benchmark=raw.median(axis=1)
f=raw.sub(benchmark,axis=0).shift(1)
res={h:[] for h in [1,3,5,10]}; ns={h:[] for h in res}
for dt in cl.index:
 for h in res:
  fut=cl.shift(-h).loc[dt]/cl.loc[dt]-1
  z=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): res[h].append(q); ns[h].append(len(z))
for h,a in res.items():
 q=pd.Series(a); print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'avgN',np.mean(ns[h]))
# regime slices for 5d
fut5=cl.shift(-5)/cl-1
for label, sl in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]:
 a=[]
 for dt in cl.index[sl]:
  z=pd.concat([f.loc[dt],fut5.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q)
 q=pd.Series(a); print('regime',label,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan)
valid=[]
for dt in cl.index:
 z=pd.concat([f.loc[dt],fut5.loc[dt]],axis=1).dropna(); valid.append(len(z))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'dates_ge8',sum(np.array(valid)>=8),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280406_relative_momentum40_signal.csv',index=False)
