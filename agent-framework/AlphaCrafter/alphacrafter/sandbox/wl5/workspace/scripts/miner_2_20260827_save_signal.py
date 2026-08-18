import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15');F={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 F[s]=-(x.close/x.open-1).ewm(span=3,min_periods=3).mean()
out=pd.DataFrame(F).sort_index();out.to_csv('factors/miner_2_intraday_reversal_20260827_signal.csv');print(out.shape,out.notna().mean().mean())
