"""Reconstruct exact signal history for admitted residual downside signed-volume pressure deceleration."""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-07-28')
def field(a, name):
    d=(pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date'])
       .drop_duplicates('date').set_index('date').sort_index())
    return d.loc[:END,name].astype(float)
p=pd.DataFrame({a:field(a,'close') for a in A})
vol=pd.DataFrame({a:field(a,'volume') for a in A})
r=p.pct_change(fill_method=None)
m=r.mean(axis=1)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var() for a in A})
e=r-beta.mul(m,axis=0)
# Exact admitted signed-volume definition.  The volume surprise remains signed:
# volume below its rolling baseline reduces pressure rather than being clipped.
vs=np.log(vol.replace(0,np.nan))-np.log(vol.replace(0,np.nan)).rolling(20,min_periods=15).mean()
pressure=(-e).clip(lower=0)*vs
signal=-(pressure.rolling(20,min_periods=12).mean()/(e.rolling(20,min_periods=15).std()+1e-12)-pressure.rolling(60,min_periods=25).mean()/(e.rolling(60,min_periods=40).std()+1e-12))
path='scripts/miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d','cutoff':str(END.date()),'signal':signal},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*len(A))
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('PER_ASSET_VALID',','.join(f'{a}:{int(signal[a].notna().sum())}' for a in A))
print('ARTIFACT',path)
