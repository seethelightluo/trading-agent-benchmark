import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 print(s, None if d is None else (len(d),str(d.date.iloc[0]),str(d.date.iloc[-1])))
