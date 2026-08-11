import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows);w=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=w.pct_change()
dx=pd.read_csv('../persistent/index_data/DXY.csv');dx['date']=pd.to_datetime(dx['date']);dx=dx.set_index('date')['close'].reindex(w.index).ffill();dr=dx.pct_change();mkt=r.mean(axis=1)
bm=pd.DataFrame(index=w.index,columns=w.columns,dtype=float);bd=bm.copy()
for s in w.columns:
 bm[s]=r[s].rolling(60).cov(mkt)/mkt.rolling(60).var();bd[s]=r[s].rolling(60).cov(dr)/dr.rolling(60).var()
# residualized 3d reversal, signal known at t from returns through t-3
raw=-w.shift(3)/w.shift(6)+1
f=raw-bm.mul(mkt.rolling(3).sum(),axis=0)-bd.mul(dr.rolling(3).sum(),axis=0);f=f.sub(f.median(axis=1),axis=0)
def calc(h):
 fut=w.shift(-h)/w-1; q=[];ns=[]
 for dt in w.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):q.append(v);ns.append(len(z))
 q=pd.Series(q);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271007_macro_residual_reversal_signal.csv',index=False)
