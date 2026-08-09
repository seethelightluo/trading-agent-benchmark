import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
END=pd.Timestamp('2026-12-17')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 prices[s]=d[d.date<=END].set_index('date').close.astype(float)
P=pd.concat(prices,axis=1); R=P.pct_change()
# Trend efficiency: signed 20-session return divided by total absolute daily movement.
# This rewards persistent directional movement while penalizing choppy paths.
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 r=R[s]
 F[s]=r.rolling(20,min_periods=20).sum()/(r.abs().rolling(20,min_periods=20).sum()+1e-12)

def calc(Y, period=None):
 vals=[]; ns=[]
 idx=F.index if period is None else F.index[(F.index>=period[0])&(F.index<=period[1])]
 for dt in idx:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=np.asarray(vals)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
for k in [1,5,10]:
 Y=P.shift(-k)/P-1
 print('horizon',k,'dates avgN IC ICIR hit',calc(Y))
for p in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]:
 print('regime',p,calc(P.shift(-1)/P-1,p))
r=F.rank(axis=1,pct=True)
print('coverage',float(F.notna().sum().sum()/F.size),'avgN',float(F.notna().sum(axis=1).mean()),'turnover',float(r.diff().abs().mean(axis=1).mean()))
F.stack().rename('factor').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261217_trend_efficiency_signal.csv',index=False)
