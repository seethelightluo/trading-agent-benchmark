import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  z=d[['date','close']].drop_duplicates('date'); z['symbol']=s; rows.append(z)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=cl.pct_change(); vol=r.rolling(20,min_periods=15).std()
shock=-(r.rolling(5,min_periods=4).sum().sub(r.rolling(5,min_periods=4).sum().median(axis=1),axis=0)).div(vol).shift(1)
disp=r.rolling(5,min_periods=4).std().mean(axis=1)
# Only scale with information known at t; high dispersion should strengthen reversal.
disp_base=disp.rolling(120,min_periods=60).median().shift(1)
disp_rank=(disp/disp_base).clip(0.5,1.5).fillna(1.0)
f=shock.mul(0.5+disp_rank, axis=0)

def calc(mask,h=1):
 a=[]
 for dt in cl.index[mask]:
  q=pd.concat([f.loc[dt],(cl.shift(-h).loc[dt]/cl.loc[dt]-1)],axis=1).dropna()
  if len(q)>=8:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x): a.append(x)
 q=pd.Series(a); return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan,(q>0).mean()
for h in [1,3,5,10]: print('horizon',h,'obs IC ICIR hit',calc(np.ones(len(cl),dtype=bool),h))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]: print('regime',name,'n IC ICIR hit',calc(mask,1))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20281005_dispersion_scaled_shock_reversal_signal.csv',index=False)
