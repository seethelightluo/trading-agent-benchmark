import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:cut]; R=P.pct_change()
def z(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
# One-day reversal plus a modest three-day reversal; all inputs are completed-day returns.
F=z(-R.rolling(1).sum())+0.50*z(-R.rolling(3).sum()); F=F.replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_3_20261217_multihorizon_reversal_signal.csv',index_label='date')
print('rows',len(F),'assets',len(U),'coverage',F.notna().sum().sum()/F.size)
print('period',F.index.min().date(),F.index.max().date())
