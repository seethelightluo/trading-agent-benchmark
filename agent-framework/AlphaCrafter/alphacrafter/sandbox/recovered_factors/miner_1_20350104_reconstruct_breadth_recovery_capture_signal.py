"""Reconstruct exact history for admitted residual breadth-recovery capture factor."""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Original revalidation used source data available through this cutoff.
END=pd.Timestamp('2027-07-28')
def close(a):
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
    return d.loc[:END,'close'].astype(float)
price=pd.DataFrame({a:close(a) for a in A})
r=price.pct_change(fill_method=None)
market=r.mean(axis=1)
# e_i,t = r_i,t - beta_i,M,60 r_M,t, with original 40-observation minimum.
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(market)/market.rolling(60,min_periods=40).var() for a in A})
residual=r-beta.mul(market,axis=0)
# Breadth shock is the positive part of the daily change in fraction positive.
breadth=(r>0).mean(axis=1)
positive_breadth_change=breadth.diff().where(lambda x:x>0,0.0)
signal=pd.DataFrame({a:residual[a].rolling(60,min_periods=40).cov(positive_breadth_change)/positive_breadth_change.rolling(60,min_periods=40).var() for a in A})
path='scripts/miner_1_breadth_recovery_capture_60d_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'miner_1_breadth_recovery_capture_60d','cutoff':str(END.date()),'signal':signal},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_1_breadth_recovery_capture_60d')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*len(A))
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('ARTIFACT',path)
