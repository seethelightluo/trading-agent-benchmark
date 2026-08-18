import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
x=get_stock_daily_data('SPX',days=3000).set_index('date').close; d=get_index_daily_data('DXY',days=3000).set_index('date').close
print(x.index.dtype,d.index.dtype,len(x.index.intersection(d.index)),x.index[:2],d.index[:2])
