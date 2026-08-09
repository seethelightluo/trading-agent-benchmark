import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill().loc[:cut]
R=P.pct_change(); disp=R.rolling(5,min_periods=5).std().mean(axis=1); q=disp.rolling(120,min_periods=60).apply(lambda x:(x<=x[-1]).mean(),raw=True)
F=-R.rolling(5,min_periods=5).sum().mul((0.5+q).clip(.5,1.5),axis=0)
F.stack().rename('signal').to_csv('scripts/miner_1_20261217_dispersion_reversal_signal.csv',header=True)
print('wrote',F.notna().sum().sum(),'signals')
