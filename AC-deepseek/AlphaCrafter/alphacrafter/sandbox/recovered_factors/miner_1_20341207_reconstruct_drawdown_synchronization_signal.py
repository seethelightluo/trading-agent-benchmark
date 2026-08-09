"""Reconstruct serialized signal history for admitted drawdown-synchronization factor.
Uses the persisted factor definition and its latest validation cutoff, without future rows.
"""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-03-21')
def close(a):
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
    return d.loc[:END,'close'].astype(float)
p=pd.DataFrame({a:close(a) for a in A})
r=p.pct_change(fill_method=None)
# Persisted expression: breadth is the share below 95% of each asset's trailing 60-day high;
# factor is 20-day improvement (lower current correlation) in 60-day corr to breadth change.
breadth=(p/p.rolling(60,min_periods=40).max()<.95).mean(axis=1).astype(float)
shock=breadth.diff()
corr=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(shock) for a in A})
factor=corr.shift(20)-corr
with open('scripts/miner_2_drawdown_synchronization_improvement_60_20_signal.pkl','wb') as fh:
    pickle.dump({'factor_id':'miner_2_drawdown_synchronization_improvement_60_20','cutoff':str(END.date()),'signal':factor},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR miner_2_drawdown_synchronization_improvement_60_20')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(factor),'CELLS',factor.shape[0]*factor.shape[1])
print('VALID_CELLS',int(factor.notna().sum().sum()),'COVERAGE',round(float(factor.notna().mean().mean()),6))
print('DATES_ANY',int(factor.notna().any(axis=1).sum()),'DATES_COMPLETE',int((factor.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',factor.dropna(how='all').index.min().date(),'LAST_VALID',factor.dropna(how='all').index.max().date())
print('ARTIFACT scripts/miner_2_drawdown_synchronization_improvement_60_20_signal.pkl')
