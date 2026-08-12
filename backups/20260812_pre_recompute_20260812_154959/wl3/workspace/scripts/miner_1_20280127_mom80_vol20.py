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
# 80-session trend normalized by recent 20-session realized volatility; lag one session.
f=((cl/cl.shift(80)-1)/r.rolling(20,min_periods=15).std()).shift(1)
def calc(h):
 fut=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(a): vals.append(a); ns.append(len(z))
 q=pd.Series(vals)
 return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns),q.tail(500).mean()
print('cutoff',cl.index.max().date(),'total_dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:
 n,ic,ir,hit,avg,recent=calc(h); print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(ir,4),'hit',round(hit,4),'avg_n',round(avg,2),'recent500',round(recent,7))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280127_mom80_vol20_signal.csv',index=False)
