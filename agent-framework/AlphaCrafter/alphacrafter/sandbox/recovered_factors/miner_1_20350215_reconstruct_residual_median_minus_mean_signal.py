"""Reconstruct signal history for admitted residual median-minus-mean factor.
Uses only same-date returns and rolling histories, limited to source availability.
"""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-07-28')
close={}
for a in A:
    d=(pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date'])
       .drop_duplicates('date').set_index('date').sort_index())
    close[a]=d.loc[:END,'close'].astype(float).replace([0,np.inf,-np.inf],np.nan)
close=pd.DataFrame(close)
r=close.pct_change(fill_method=None)
# Persisted definition: e_i=r_i-beta_i*M, M is contemporaneous equal-weight
# cross-asset return, beta is a 60-observation rolling covariance ratio.
m=r.mean(axis=1,skipna=True)
beta=r.rolling(60,min_periods=42).cov(m).div(m.rolling(60,min_periods=42).var(),axis=0)
e=r-beta.mul(m,axis=0)
signal=(e.rolling(60,min_periods=42).median()-e.rolling(60,min_periods=42).mean()).div(e.rolling(60,min_periods=42).std())
signal=signal.replace([np.inf,-np.inf],np.nan)
path='scripts/miner_3_residual_median_minus_mean_60d_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'miner_3_residual_median_minus_mean_60d','cutoff':str(END.date()),'signal':signal},fh,pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_3_residual_median_minus_mean_60d')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*len(A))
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('PER_ASSET_VALID',','.join(f'{a}:{int(signal[a].notna().sum())}' for a in A))
print('ARTIFACT',path)
