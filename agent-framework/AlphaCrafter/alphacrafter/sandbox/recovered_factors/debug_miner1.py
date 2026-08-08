from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
import pandas as pd
for a in ['SPX','CN10Y']:
 d=get_stock_daily_data(a,5000);print(a,d.shape,d.columns.tolist(),d.head(2),d.tail(2),d.date.dtype)
