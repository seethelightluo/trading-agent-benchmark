import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:get_stock_daily_data(s,days=3600).set_index('date')['close'] for s in U}).sort_index().ffill();r=px.pct_change();low=px.rolling(60,min_periods=40).min();rec=px/low-1;mom=px.pct_change(20);down=r.where(r<0).rolling(40,min_periods=25).std();f=rec*(1+mom.clip(lower=-.5))/(down*np.sqrt(252));print(px.shape, f.notna().sum(axis=1).describe());print(f.tail());print(f.index[-1])
