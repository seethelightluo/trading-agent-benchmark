import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 D[s]=x[x.date<=END].set_index('date').close.astype(float)
P=pd.concat(D,axis=1,sort=True); R=P.pct_change()
w=60
F=-R.rolling(w,min_periods=w).sum()/R.rolling(w,min_periods=w).std().replace(0,np.nan)
Y=P.shift(-1)/P-1

def ev(idx):
 q=[]; ns=[]
 for dt in idx:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.asarray(q); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('full',ev(F.index))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]: print('regime',a,b,ev(F.index[(F.index>=a)&(F.index<=b)]))
rank=F.rank(axis=1,pct=True); print('coverage',F.notna().sum().sum()/F.size,'avgN',F.notna().sum(axis=1).mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
F.stack().rename('factor').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261217_volscaled60_signal.csv',index=False)
