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
# Novel interpretable factor: 5-day reversal scaled by asset volatility,
# activated when cross-asset dispersion is high (dispersion relative to its 60d median).
ret5=cl.pct_change(5); vol20=r.rolling(20).std(); disp=r.std(axis=1).rolling(20).mean(); med=disp.rolling(60).median()
state=(disp/med).clip(0.5,2.0)
f=(-ret5/vol20).mul(state,axis=0).shift(1)
def calc(h,mask=None):
 a=[]
 idx=cl.index if mask is None else cl.index[mask]
 for dt in idx:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): a.append(z)
 return pd.Series(a)
for h in [1,3,5,10]:
 q=calc(h); print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:
 q=calc(5,mask); print('regime',name,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20281005_dispersion_state_reversal_signal.csv',index=False)
