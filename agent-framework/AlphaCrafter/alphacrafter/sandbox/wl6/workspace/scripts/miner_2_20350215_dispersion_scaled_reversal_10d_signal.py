import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
raw=-P.pct_change(10)/(r.rolling(20).std()*np.sqrt(252))
disp=r.std(axis=1).rolling(20).rank(pct=True)
f=raw.where(disp>0.5).shift(1)
f.index.name='date'; f.to_csv('scripts/miner_2_20350215_dispersion_scaled_reversal_10d_signal.csv')
print('saved',f.shape,'valid_fraction',f.notna().sum().sum()/(f.shape[0]*15))
