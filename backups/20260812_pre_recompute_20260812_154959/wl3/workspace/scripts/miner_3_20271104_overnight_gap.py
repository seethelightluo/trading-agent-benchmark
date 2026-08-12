import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close','open']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
p=pd.concat(rows)
cl=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill()
op=p.pivot(index='date',columns='symbol',values='open').reindex(cl.index).ffill()
# Overnight gap reversal: prior close-to-open shock, lagged for availability.
gap=op/cl.shift(1)-1
sig=(-gap).rolling(3,min_periods=3).mean().shift(1)

def calc(h):
 fut=cl.shift(-h)/cl-1; vals=[]; ns=[]; dates=[]
 for dt in cl.index:
  z=pd.concat([sig.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(r): vals.append(r); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=pd.to_datetime(dates));
 return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]: print('H',h,calc(h))
# regime split for daily horizon
f=[]
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-12-31')]:
 x=sig.loc[a:b]; y=(cl.shift(-1)/cl-1).loc[a:b]; z=[]
 for dt in x.index:
  w=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(w)>=8:z.append(spearmanr(w.iloc[:,0],w.iloc[:,1]).statistic)
 f.append((a,b,len(z),np.nanmean(z),np.nanstd(z,ddof=1)))
print('regimes',f)
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271104_overnight_gap_signal.csv',index=False)
