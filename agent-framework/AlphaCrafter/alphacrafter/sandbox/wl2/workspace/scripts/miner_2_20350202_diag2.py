import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; o={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 o[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(o).sort_index(); r=p.pct_change(); neg=r.shift(1).where(r.shift(1)<0).rolling(30,min_periods=20).std(); rec=p.shift(1)/p.shift(1).rolling(60,min_periods=40).min()-1
print('rec',rec.notna().sum().sum(),'neg',neg.notna().sum().sum(),'both',(rec/neg).notna().sum().sum());print(neg.iloc[-1])
