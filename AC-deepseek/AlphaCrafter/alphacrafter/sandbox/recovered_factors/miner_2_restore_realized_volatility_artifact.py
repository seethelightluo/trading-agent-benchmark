"""Restore reproducible signal artifact for admitted realized-volatility factor."""
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
    return (pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])
            .drop_duplicates('date').set_index('date').sort_index().close.astype(float))
p=pd.DataFrame({a:load(a) for a in A}).sort_index()
s=p.pct_change().rolling(20,min_periods=15).std()
s.to_pickle('scripts/miner_2_20260716_realized_volatility_20obs_signal.pkl')
print('RESTORED',s.shape,s.index.min().date(),s.index.max().date(),float(s.notna().mean().mean()))
