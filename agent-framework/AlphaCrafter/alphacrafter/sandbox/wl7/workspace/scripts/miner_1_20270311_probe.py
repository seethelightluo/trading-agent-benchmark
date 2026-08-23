from alphacrafter.sim.utils import get_stock_daily_data
for s in ['SPX','BTC','000300.SH']:
 try:
  x=get_stock_daily_data(s,5);print(s,x, None if x is None else x.dtypes)
 except Exception as e: print(s,type(e),e)
