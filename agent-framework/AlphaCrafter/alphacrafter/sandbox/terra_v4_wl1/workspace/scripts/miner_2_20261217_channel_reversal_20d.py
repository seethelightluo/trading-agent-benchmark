import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'] for s in U},axis=1).sort_index()
W=20; hi=P.rolling(W,min_periods=15).max(); lo=P.rolling(W,min_periods=15).min(); F=-(P-lo)/(hi-lo).replace(0,np.nan)
def calc(k,period=None):
 Y=P.shift(-k)/P-1; vals=[]; ns=[]; idx=F.index if period is None else F.index[(F.index>=period[0])&(F.index<=period[1])]
 for dt in idx:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.array(vals); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),(a>0).mean()
for k in [1,5,10]: print('horizon',k,calc(k))
for p in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]: print('regime',p,calc(1,p))
r=F.rank(axis=1,pct=True); print('coverage',F.notna().sum().sum()/F.size,'avgN',F.notna().sum(axis=1).mean(),'turnover',r.diff().abs().mean(axis=1).mean())
F.stack().rename('factor').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261217_channel_reversal_20d_signal.csv',index=False)
