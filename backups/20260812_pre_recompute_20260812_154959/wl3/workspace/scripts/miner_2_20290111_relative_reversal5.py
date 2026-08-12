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
r5=cl.pct_change(5)
# Relative short-term reversal: reverse each asset's 5d move after removing the same-day cross-sectional median.
f=-(r5.sub(r5.median(axis=1),axis=0)).shift(1)
def calc(h,mask=None):
 vals=[]; ix=cl.index if mask is None else cl.index[mask]
 for dt in ix:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z)
 a=pd.Series(vals)
 return len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()
for h in [1,3,5,10]: print('horizon',h,'obs IC ICIR hit',calc(h))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026-27',(cl.index>='2026-01-01')&(cl.index<'2028-01-01')),('2028',cl.index>='2028-01-01'),('recent120',cl.index>=cl.index[-120])]: print('regime',name,calc(5,mask))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290111_relative_reversal5_signal.csv',index=False)
