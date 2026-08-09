import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}; rets={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date');d=d[d.date<=END].set_index('date'); prices[s]=d.close.astype(float); rets[s]=d.close.astype(float).pct_change()
P=pd.concat(prices,axis=1);R=pd.concat(rets,axis=1)
# 5d reversal, scaled by the asset's 20d volatility and its 20/60 volatility shock.
F=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U:
 r=R[s];v20=r.rolling(20,min_periods=15).std()*np.sqrt(20);v60=r.rolling(60,min_periods=40).std()*np.sqrt(60)
 F[s]=(-r.rolling(5,min_periods=5).sum()/v20)*(v20/v60).clip(.5,2.5)
def calc(Y,period=None):
 a=[];ns=[];idx=F.index if period is None else F.index[(F.index>=period[0])&(F.index<=period[1])]
 for dt in idx:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.asarray(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for k in [1,5,10]:
 # forward k-session simple return per asset
 Y=pd.DataFrame({s:prices[s].shift(-k)/prices[s]-1 for s in U}); print('horizon',k,'dates avgN IC ICIR hit',calc(Y))
for p in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]:print('regime',p,calc(pd.DataFrame({s:prices[s].shift(-1)/prices[s]-1 for s in U}),p))
r=F.rank(axis=1,pct=True);print('coverage',F.notna().sum().sum()/F.size,'turnover',r.diff().abs().mean(axis=1).mean(),'avgN',F.notna().sum(axis=1).mean())
F.stack().rename('factor').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261217_volshock_reversal_signal.csv',index=False)
