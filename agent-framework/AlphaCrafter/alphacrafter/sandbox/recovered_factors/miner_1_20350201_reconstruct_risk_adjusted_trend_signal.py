"""Reconstruct canonical signal panel for admitted 20-day risk-adjusted trend."""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-07-28')
close={}
for a in A:
    d=(pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date'])
       .drop_duplicates('date').set_index('date').sort_index())
    close[a]=d.loc[:END,'close'].astype(float).replace([np.inf,-np.inf,0],np.nan)
close=pd.DataFrame(close)
# Exact persisted definition: trailing compounded 20-observation return / trailing
# 20-observation standard deviation of one-day returns, with >=15 vol observations.
r=close.pct_change(fill_method=None)
signal=(close/close.shift(20)-1.0)/r.rolling(20,min_periods=15).std()
signal=signal.replace([np.inf,-np.inf],np.nan)
path='scripts/miner_3_risk_adjusted_trend_20d_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'miner_3_risk_adjusted_trend_20d','cutoff':str(END.date()),'signal':signal},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_3_risk_adjusted_trend_20d')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*len(A))
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('PER_ASSET_VALID',','.join(f'{a}:{int(signal[a].notna().sum())}' for a in A))
print('ARTIFACT',path)
