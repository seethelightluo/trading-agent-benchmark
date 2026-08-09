import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); return d.close.astype(float).loc[:'2026-07-15']
P=pd.DataFrame({s:load(s) for s in U}); R=P.pct_change(fill_method=None); vol=R.rolling(20,min_periods=15).std(); f=R.rolling(5,min_periods=5).sum()/vol
print(P.shape,P.index.min(),P.index.max(),'R valid',R.notna().sum().min(),'f valid',f.notna().sum().min(), 'rows',f.dropna(how='all').shape)
print(P.tail())
