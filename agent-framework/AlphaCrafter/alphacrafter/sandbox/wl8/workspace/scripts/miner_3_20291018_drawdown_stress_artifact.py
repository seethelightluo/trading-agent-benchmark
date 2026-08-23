import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2029-10-17']
r=P.pct_change(); trough=P.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std()
f=(-np.log(P/trough)/vol.replace(0,np.nan)).shift(1)
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20291018_drawdown_stress_signal.csv',index=False)
