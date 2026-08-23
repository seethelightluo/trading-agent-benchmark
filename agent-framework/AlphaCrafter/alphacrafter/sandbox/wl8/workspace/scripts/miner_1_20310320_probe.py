from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
for s in ['SPX','BTC','000300.SH']:
 d=get_stock_daily_data(s,days=3000); print(s,'stock',None if d is None else len(d),None if d is None else d.date.min(),None if d is None else d.date.max())
 d=get_index_daily_data(s,days=3000); print(s,'index',None if d is None else len(d),None if d is None else d.date.min(),None if d is None else d.date.max())
