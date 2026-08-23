import pandas as pd, numpy as np
p=pd.DataFrame()
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();p[s]=d.close.astype(float)
r=p.pct_change();v=r.rolling(20,min_periods=15).std()
sig=-(0.6*p.pct_change(5).shift(1)/v.shift(1)+0.4*p.pct_change(20).shift(1)/v.shift(1))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291018_blended_reversal_signal.csv',index=False)
