import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
for s in ['SPX','BTC','000300.SH']:
 d=get_stock_daily_data(s,days=3000); print(s,len(d),d.date.iloc[0],d.date.iloc[-1])
d=get_index_daily_data('DXY',days=3000); print('DXY',d.date.iloc[0],d.date.iloc[-1])
