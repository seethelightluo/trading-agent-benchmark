import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); close={}; intr={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); close[s]=x.close.astype(float).pct_change(); intr[s]=x.close.astype(float)/x.open.astype(float)-1
R=pd.concat(close,axis=1).sort_index(); I=pd.concat(intr,axis=1).sort_index()
def ev(F,h,start=None,end=None):
 Y=sum(R.shift(-k) for k in range(1,h+1)); q=[]; ns=[]; ds=[]
 for d in F.index:
  if start and d<pd.Timestamp(start): continue
  if end and d>pd.Timestamp(end): continue
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: q.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(d)
 q=pd.Series(q,index=ds); return q.mean(),q.mean()/q.std(ddof=1),len(q),np.mean(ns),(q>0).mean()
F=-I.rolling(3,min_periods=3).mean(); print('idea=intraday_body3_reversal')
for h in [1,5,10]: print(h,ev(F,h))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026')]: print(a,b,ev(F,1,a,b))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean()); F.to_csv('scripts/miner_1_20261217_intraday_body3_reversal_signal.csv')
