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
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
# Recovery-adjusted trend: 60-session return divided by the magnitude of the worst close-to-close drawdown.
wealth=(1+r.fillna(0)).rolling(60,min_periods=45).apply(np.prod,raw=True)-1
rollmax=cl.rolling(60,min_periods=45).max(); dd=cl/rollmax-1
mdd=dd.rolling(60,min_periods=45).min().abs()
f=(wealth/(mdd+0.05)).shift(1)
all_ic={}
for h in [1,3,5,10]:
 a=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v); ns.append(len(z))
 q=pd.Series(a); all_ic[h]=q
 print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'avgN',np.mean(ns))
for label,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]:
 a=[]
 for dt in cl.index[mask]:
  z=pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v)
 q=pd.Series(a); print('regime',label,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
valid=[]
for dt in cl.index: valid.append(pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna().shape[0])
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'dates_ge8',sum(np.array(valid)>=8),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280504_recovery_trend_signal.csv',index=False)
