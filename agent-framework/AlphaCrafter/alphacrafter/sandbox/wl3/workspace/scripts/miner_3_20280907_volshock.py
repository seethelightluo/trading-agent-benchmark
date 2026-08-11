import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
# Volatility shock reversal: lower recent volatility relative to long baseline is expected to precede better returns.
v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=40).std()
f=-(v10/v60-1).shift(1)
allh={}
for h in [1,3,5,10]:
 a=[]
 for dt in cl.index:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z):a.append(z)
 q=pd.Series(a);allh[h]=(len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean())
 print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:
 a=[]
 for dt in cl.index[mask]:
  q=pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('regime',name,'n',len(a),'IC',np.mean(a) if a else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20280907_volshock_signal.csv',index=False)
