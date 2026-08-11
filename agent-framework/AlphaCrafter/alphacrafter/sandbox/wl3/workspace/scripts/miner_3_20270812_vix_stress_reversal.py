import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows);w=pivot=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=w.pct_change()
eq=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']; eqr=r[eq].mean(axis=1)
# Observation-only VIX, lagged/realized only; elevated/rising volatility conditions
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(w.index).ffill()
vshock=v.pct_change(5).clip(-1,1); vlevel=(v/v.rolling(60,min_periods=30).median()-1).clip(-1,2)
ret5=w/w.shift(5)-1; eq5=eqr.rolling(5,min_periods=4).sum(); cov=r.rolling(30,min_periods=20).cov(eqr); var=eqr.rolling(30,min_periods=20).var(); beta=cov.div(var,axis=0); residual=ret5-beta.mul(eq5,axis=0)
# contrarian residual, with a transparent VIX stress multiplier, all inputs known at date
stress=(1+0.6*vshock.clip(lower=0)+0.4*vlevel.clip(lower=0)).clip(1,2.5)
f=-residual.mul(stress,axis=0); f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1; qs=[]; ns=[]; idx=w.index
 if a: idx=idx[(idx>=a)&(idx<=b)]
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
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270812_vix_stress_reversal_signal.csv',index=False)
