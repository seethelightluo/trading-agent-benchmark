import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); p[s]=d.sort_values('date').set_index('date').close.astype(float)
pd.concat(p,axis=1).sort_index().ffill().pct_change(20).shift(1).to_csv('scripts/miner_1_20330624_breadth_trend_signal.csv')
