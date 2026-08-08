"""Reconstruct exact visible-data history for market synchronization admitted factor."""
import pickle
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-03-21')
def close(a):
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
    return d.loc[:END,'close'].astype(float)
price=pd.DataFrame({a:close(a) for a in A})
ret=price.pct_change(fill_method=None)
# Persisted definition: trailing 60-session correlation to equal-weight market,
# less its value twenty sessions ago.  The mean includes all 15 tradables.
market=ret.mean(axis=1)
corr=pd.DataFrame({a:ret[a].rolling(60,min_periods=40).corr(market) for a in A})
signal=corr-corr.shift(20)
path='scripts/miner_2_market_synchronization_increase_60_20_signal.pkl'
with open(path,'wb') as fh:
    pickle.dump({'factor_id':'market_synchronization_increase_60_20','cutoff':str(END.date()),'signal':signal},fh,protocol=pickle.HIGHEST_PROTOCOL)
print('FACTOR market_synchronization_increase_60_20')
print('CUTOFF',END.date(),'UNIVERSE',len(A),'DATES',len(signal),'CELLS',signal.shape[0]*signal.shape[1])
print('VALID_CELLS',int(signal.notna().sum().sum()),'COVERAGE',round(float(signal.notna().mean().mean()),6))
print('DATES_ANY',int(signal.notna().any(axis=1).sum()),'DATES_COMPLETE',int((signal.notna().sum(axis=1)==len(A)).sum()))
print('FIRST_VALID',signal.dropna(how='all').index.min().date(),'LAST_VALID',signal.dropna(how='all').index.max().date())
print('ARTIFACT',path)
