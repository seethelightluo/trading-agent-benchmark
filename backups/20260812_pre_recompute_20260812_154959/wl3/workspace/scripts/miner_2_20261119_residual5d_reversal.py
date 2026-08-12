import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2200)
 if d is None or len(d)<150: d=get_index_daily_data(s,2200)
 if d is not None:
  x=d[['date','close']].copy(); x['symbol']=s; rows.append(x)
w=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index()
r=w.pct_change(); m=r.mean(axis=1)
# residual 5-day move versus contemporaneous cross-asset mean, normalized by idiosyncratic 20d vol
res=r.sub(m,axis=0)
res5=res.rolling(5,min_periods=5).sum(); vol=res.rolling(20,min_periods=12).std()*np.sqrt(20)
f=(-res5/vol.replace(0,np.nan)).clip(-4,4).replace([np.inf,-np.inf],np.nan)
def calc(h):
 fut=w.shift(-h)/w-1; q=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q).dropna();return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]:print('H',h,'n mean std ICIR hit avgN',calc(h))
ranks=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'rank_turnover',((ranks-ranks.shift()).abs().mean(axis=1)).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=[]; fut=w.shift(-1)/w-1
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('REG',lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20261119_residual5d_reversal_signal.csv',index=False)
