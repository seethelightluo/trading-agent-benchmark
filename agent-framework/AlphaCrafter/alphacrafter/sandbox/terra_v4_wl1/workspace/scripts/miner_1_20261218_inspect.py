import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']); print(s,len(d),d.date.min(),d.date.max(),d.close.notna().sum())
