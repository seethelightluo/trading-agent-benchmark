import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
# Tail-risk-adjusted momentum: medium return penalized by downside volatility, lagged to prevent look-ahead.
down=r.where(r<0,0.0)
f=(r.rolling(30,min_periods=20).sum()/(-down.pow(2).rolling(60,min_periods=30).mean().pow(0.5))).shift(1)
def calc(h):
 fut=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(a): vals.append(a); ns.append(len(z))
 q=pd.Series(vals)
 return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'total_dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:
 n,ic,ir,hit,avg=calc(h); print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(ir,4),'hit',round(hit,4),'avg_n',round(avg,2))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20271230_tailrisk_momentum_signal.csv',index=False)
# regimes
fut=cl.shift(-1)/cl-1; q=[]
for dt in cl.index:
 z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(q,columns=['date','ic']).set_index('date')
for name,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025+','2025','2099')]:
 x=q.loc[(q.index>=a)&(q.index<=b),'ic']; print('REG',name,'n',len(x),'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(len(x)),4))
