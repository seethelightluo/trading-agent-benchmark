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
p=pd.concat(rows); wide=p.pivot(index='date',columns='symbol',values='close').sort_index(); r=wide.pct_change()
disp=r.abs().mean(axis=1); med=disp.rolling(60,min_periods=40).median(); vol=r.rolling(20,min_periods=12).std()*np.sqrt(20)
# Three-session reversal, normalized by volatility and smoothly conditioned on cross-asset dispersion.
f=(-wide.pct_change(3)/vol)*(disp/med).clip(.5,2).values[:,None]; f=f.replace([np.inf,-np.inf],np.nan)
def calc(h):
 fut=wide.shift(-h)/wide-1; q=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q).dropna(); return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',wide.index.max().date(),'dates',len(wide),'instruments',len(wide.columns))
for h in [1,3,5,10]:print('H',h,'n IC std ICIR hit avgN',calc(h))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum())
rank=f.rank(axis=1,pct=True);print('rank_turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(wide.shift(-1)/wide-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('REG',a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20261119_dispersion_scaled_reversal_3d_signal.csv',index=False)
