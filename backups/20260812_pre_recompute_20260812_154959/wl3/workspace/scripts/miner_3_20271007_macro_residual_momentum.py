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
try:
 dx=pd.read_csv('../persistent/index_data/DXY.csv'); dx['date']=pd.to_datetime(dx['date']); dx=dx.set_index('date')['close'].reindex(w.index).ffill(); dr=dx.pct_change()
except Exception: dr=pd.Series(0.,index=w.index)
raw=w.shift(3)/w.shift(18)-1;mkt=r.mean(axis=1); mkt15=mkt.rolling(15).sum()
# align explicit rolling covariance via per-column rolling correlation regression
betam=pd.DataFrame(index=w.index,columns=w.columns,dtype=float); betad=betam.copy()
for s in w.columns:
 betam[s]=r[s].rolling(60).cov(mkt)/mkt.rolling(60).var()
 betad[s]=r[s].rolling(60).cov(dr)/dr.rolling(60).var()
f=raw-betam.mul(mkt15,axis=0).sub(betad.mul(dr.rolling(15).sum(),axis=0),fill_value=0)
f=f.sub(f.median(axis=1),axis=0)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1;qs=[];ns=[];idx=w.index
 if a:idx=idx[(idx>=a)&(idx<=b)]
 for dt in idx:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]: print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(w.index.max().date()))]: print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271007_macro_residual_momentum_signal.csv',index=False)
