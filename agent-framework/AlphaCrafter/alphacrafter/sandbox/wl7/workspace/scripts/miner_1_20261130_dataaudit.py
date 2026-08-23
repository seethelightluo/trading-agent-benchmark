import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 try:d=get_index_daily_data(s,days=2600)
 except Exception:d=get_stock_daily_data(s,days=2600)
 print(s,len(d),d.date.iloc[0],d.date.iloc[-1],d.close.notna().sum())
