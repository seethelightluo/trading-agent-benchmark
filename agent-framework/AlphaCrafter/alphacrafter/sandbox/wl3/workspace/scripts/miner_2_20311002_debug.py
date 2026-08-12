from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
for s in ['SPX','BTC']:
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000); print(s,fn.__name__, None if x is None else (len(x),x.columns.tolist(),x.head(1).to_dict('records')))
  except Exception as e: print('err',s,fn.__name__,repr(e))
