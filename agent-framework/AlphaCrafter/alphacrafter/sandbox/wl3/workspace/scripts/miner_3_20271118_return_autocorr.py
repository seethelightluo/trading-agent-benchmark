import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
pd0=pd.concat(rows); cl=pd0.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
def ac(x):
 a=x[:-1]; b=x[1:]
 if len(a)<8 or a.std()==0 or b.std()==0:return np.nan
 return np.corrcoef(a,b)[0,1]
f=r.rolling(20,min_periods=15).apply(ac,raw=True).shift(1)
def calc(h):
 fut=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); ns.append(len(z))
 q=pd.Series(vals); sd=q.std(ddof=1)
 return len(q),q.mean(),sd,q.mean()/sd*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'total_dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:
 n,ic,sd,icir,hit,avg=calc(h);print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(icir,4),'hit',round(hit,4),'avg_n',round(avg,2))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271118_return_autocorr_signal.csv',index=False)
