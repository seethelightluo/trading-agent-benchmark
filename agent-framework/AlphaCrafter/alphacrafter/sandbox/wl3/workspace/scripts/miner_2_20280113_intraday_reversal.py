import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
a=pd.concat(rows).pivot(index='date',columns='symbol',values='open').sort_index().ffill(); c=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
# Lagged intraday reversal: prior session open-to-close loss is bullish for next session; normalize by 20d volatility.
r=c.pct_change(); intr=c/a-1
f=(-intr/r.rolling(20,min_periods=15).std()).shift(1)
def calc(h):
 fut=c.shift(-h)/c-1; vals=[];ns=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 q=pd.Series(vals);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'total_dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]:
 n,ic,ir,hit,avg=calc(h);print('H',h,'obs',n,'IC',round(ic,7),'ICIR',round(ir,4),'hit',round(hit,4),'avg_n',round(avg,2))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20280113_intraday_reversal_signal.csv',index=False)
