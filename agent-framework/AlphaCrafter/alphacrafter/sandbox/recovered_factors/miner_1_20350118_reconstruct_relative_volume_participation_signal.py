"""Reconstruct canonical signal panel for admitted relative-volume participation factor."""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Shared historical data endpoint used by prior reconstruction cycles; no post-cutoff data.
END=pd.Timestamp('2027-07-28')
vol={}
for a in A:
    d=(pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date'])
       .drop_duplicates('date').set_index('date').sort_index())
    vol[a]=d.loc[:END,'volume'].astype(float).replace(0,np.nan)
volume=pd.DataFrame(vol)
# Exact admitted definition, with the producer's 15-observation availability rule.
signal=np.log(volume / volume.rolling(20,min_periods=15).mean())
path='scripts/miner_3_relative_volume_participation_20d_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'miner_3_relative_volume_participation_20d',
                 'cutoff':str(END.date()),'signal':signal},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_3_relative_volume_participation_20d')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*len(A))
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('PER_ASSET_VALID',','.join(f'{a}:{int(signal[a].notna().sum())}' for a in A))
print('ARTIFACT',path)
