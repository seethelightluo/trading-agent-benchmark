import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
c=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=c.pct_change()
# Lagged realized return asymmetry: negative skewness of recent returns is a risk signal;
# contrarian score rewards positive skew / penalizes left-tail behavior, normalized cross-sectionally.
f=r.rolling(20,min_periods=15).skew().shift(1)
def calc(h):
 fut=c.shift(-h)/c-1;v=[];ns=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):v.append(q);ns.append(len(z))
 q=pd.Series(v);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'total_dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]:
 n,ic,ir,hit,avg=calc(h);print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(ir,4),'hit',round(hit,4),'avg_n',round(avg,2))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20280113_skew_signal.csv',index=False)
