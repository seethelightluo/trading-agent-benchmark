import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000);x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff();trend=np.log(p/p.shift(60));vol=r.rolling(20,min_periods=15).std();f=(-np.log(p/p.shift(5))*(1+0.5*(trend.abs()/(vol*np.sqrt(60))).clip(0,3))).shift(1);f.to_csv('scripts/miner_3_20311030_shock_reversal_signal.csv',index_label='date');print('saved',f.shape,'coverage',f.notna().sum().mean()/15)
