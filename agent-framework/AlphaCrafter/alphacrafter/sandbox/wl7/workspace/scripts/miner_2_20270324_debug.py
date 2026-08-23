import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={}
for s in U:
 d=get_stock_daily_data(s,5000)
 print(s, None if d is None else (len(d),str(d.date.min()),str(d.date.max())))
 if d is not None: ds[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(ds).sort_index(); print('px',len(px),px.notna().sum().to_dict())
r=px.pct_change(); down=r.where(r<0).rolling(60,min_periods=30).std(); f=(px/px.shift(20)-1).shift(1)/down.shift(1); print('f valid',f.notna().sum().to_dict(),len(f.dropna(how='all')))
