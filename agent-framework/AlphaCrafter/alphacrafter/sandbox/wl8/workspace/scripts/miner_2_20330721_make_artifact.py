import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
f=(p.shift(1)/p.shift(21)-1)/vol.shift(1); f=f.sub(f.median(axis=1),axis=0).rolling(10,min_periods=10).mean(); f.to_csv('scripts/miner_2_20330721_volnorm_relative_strength_smooth10_signal.csv')
