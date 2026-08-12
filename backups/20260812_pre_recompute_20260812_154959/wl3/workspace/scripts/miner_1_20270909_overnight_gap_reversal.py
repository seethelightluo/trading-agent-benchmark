import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','close','high','low']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
p=pd.concat(rows)
q={z:p.pivot(index='date',columns='symbol',values=z).sort_index().ffill() for z in ['open','close','high','low']}; c,o,h,l=q['close'],q['open'],q['high'],q['low']
# Overnight gap, normalized by recent true range; reversal is expected next-day response.
prev=c.shift(1); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).groupby(level=0,axis=1).max() if False else (pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],keys=['a','b','d']).groupby(level=1).max())
atr=tr.rolling(14,min_periods=10).mean()
gap=(o/prev-1)/atr.replace(0,np.nan)
# Damp extreme gaps and remove daily cross-sectional location.
f=(-np.tanh(gap/2.0)).sub((-np.tanh(gap/2.0)).median(axis=1),axis=0).clip(-2,2)
def calc(horizon,ix=None):
 fut=c.shift(-horizon)/c-1; vals=[]; ns=[]
 for dt in (c.index if ix is None else ix):
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(r): vals.append(r); ns.append(len(z))
 s=pd.Series(vals); return len(s),s.mean(),s.std(ddof=1),s.mean()/s.std(ddof=1)*np.sqrt(len(s)),(s>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'dates',len(c),'instruments',len(c.columns))
for k in [1,3,5,10]: print('H',k,calc(k))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(c.index.max().date()))]:
 print('REG',a,b,calc(1,c.loc[a:b].index))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270909_overnight_gap_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
