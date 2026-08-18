import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
for s in ['SPX','XAU']:
 x=get_stock_daily_data(s,20)
 print(s, x is None, x.head() if x is not None else None, x.dtypes if x is not None else '')
