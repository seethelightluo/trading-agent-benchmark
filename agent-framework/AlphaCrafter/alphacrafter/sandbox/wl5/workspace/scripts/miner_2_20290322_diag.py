import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 d=get_stock_daily_data(s,days=2400)
 print(s, None if d is None else (len(d), d.date.iloc[0] if len(d) else None, d.date.iloc[-1] if len(d) else None))
D={s:get_stock_daily_data(s,days=2400).set_index('date')['close'].astype(float) for s in U if get_stock_daily_data(s,days=2400) is not None}
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); print('p',p.shape,'rvalid',r.notna().sum().to_dict())
print('rv',r.rolling(20).std().notna().sum().sum())
