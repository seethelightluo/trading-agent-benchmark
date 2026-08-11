from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 d=get_stock_daily_data(s,days=4000); print(s, None if d is None else len(d), None if d is None else (d['date'].min(),d['date'].max()))
