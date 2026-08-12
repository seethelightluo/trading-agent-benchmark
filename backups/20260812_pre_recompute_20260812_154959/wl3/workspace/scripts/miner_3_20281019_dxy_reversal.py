import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
m=pd.read_csv('../persistent/index_data/DXY.csv');m['date']=pd.to_datetime(m['date']); col='close' if 'close' in m.columns else [c for c in m if c!='date'][0]
dxy=m.set_index('date')[col].reindex(cl.index).ffill(); z=(dxy-dxy.rolling(60).mean())/dxy.rolling(60).std()
# Reversal is strengthened when dollar is unusually strong, a cross-asset risk-off regime
f=(-cl.pct_change(5)).mul(1+0.75*np.clip(z,0,2),axis=0).shift(1)
def calc(h,mask=None):
 a=[]
 ix=cl.index if mask is None else cl.index[mask]
 for dt in ix:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   r=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(r):a.append(r)
 q=pd.Series(a);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()
for h in [1,3,5,10]:print('horizon',h,'obs IC ICIR hit',calc(h))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:print('regime',name,calc(5,mask))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20281019_dxy_reversal_signal.csv',index=False)
