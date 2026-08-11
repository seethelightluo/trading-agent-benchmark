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
r=cl.pct_change(); mkt=r.mean(axis=1)
# 5d asset return minus contemporaneous cross-asset mean, scaled by trailing idiosyncratic volatility; lag one day
raw=r.rolling(5).sum().sub(mkt.rolling(5).sum(),axis=0)
idvol=r.sub(mkt,axis=0).rolling(20,min_periods=15).std()
f=(-raw/idvol.replace(0,np.nan)).shift(1)
decay={1:[],3:[],5:[],10:[]}; ns=[]
for dt in cl.index:
 for h in decay:
  fut=cl.shift(-h).loc[dt]/cl.loc[dt]-1
  z=pd.concat([f.loc[dt],fut],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): decay[h].append(q)
for h,a in decay.items():
 q=pd.Series(a); print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
q=pd.Series(decay[5]); valid=f.notna(); turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',valid.sum(axis=1).mean(),'coverage',valid.mean().mean(),'turnover',turn)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280309_idio_reversal_signal.csv',index=False)
