import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 return d
def build(look):
 rows=[]
 for s in U:
  d=fetch(s)
  if d is None:continue
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();v=d.close.pct_change().rolling(20,min_periods=15).std()
  f=-(d.close.pct_change(look)/(v*np.sqrt(look))).replace([np.inf,-np.inf],np.nan);r=d.close.shift(-1)/d.close-1
  q=pd.DataFrame({'factor':f,'forward_return_1d':r}).dropna().reset_index();q['symbol']=s;rows.append(q)
 return pd.concat(rows,ignore_index=True)
for look in [2,3]:
 q=build(look);q.to_csv(f'scripts/miner_3_20281130_volscaled_reversal{look}_signal.csv',index=False);print(look,len(q),q.symbol.nunique(),q.date.nunique())
