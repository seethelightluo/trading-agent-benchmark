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
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change(); bench=r.mean(axis=1)
def evaluate(f,h=1):
 fut=cl.shift(-h)/cl-1; qs=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): qs.append(q);ns.append(len(z))
 q=pd.Series(qs); return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns))
for w in [40,60,90]:
 cov=r.rolling(w,min_periods=max(20,w//2)).cov(bench); var=bench.rolling(w,min_periods=max(20,w//2)).var(); beta=cov.div(var,axis=0)
 f=(-(r.rolling(5,min_periods=5).sum()-beta.mul(bench.rolling(5,min_periods=5).sum(),axis=0))).shift(1)
 print('window',w,'metrics',evaluate(f),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
 out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv(f'scripts/miner_3_20280113_beta{w}_signal.csv',index=False)
