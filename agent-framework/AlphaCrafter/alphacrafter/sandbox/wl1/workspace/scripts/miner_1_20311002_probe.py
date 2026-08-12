from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in U:
 try:
  d=get_stock_daily_data(s,5000)
  print(s, None if d is None else (len(d),list(d.columns),d.close.notna().sum() if d is not None else 0))
 except Exception as e: print(s,'ERR',type(e).__name__,str(e))
