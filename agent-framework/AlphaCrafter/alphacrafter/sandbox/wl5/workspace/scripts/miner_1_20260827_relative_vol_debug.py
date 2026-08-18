import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,3000).copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index();r=P.pct_change(fill_method=None);v=r.rolling(20,min_periods=15).std();f=-v.div(v.median(axis=1),axis=0);fr=P.pct_change(fill_method=None).shift(-1)
print(f.notna().sum(axis=1).value_counts().sort_index().tail());print(fr.notna().sum(axis=1).value_counts().sort_index().tail());print(pd.concat([f,fr],axis=1).dropna().shape)
