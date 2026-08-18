import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>150:return d
  except: pass
xs={s:fetch(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float)
gate=(v>v.rolling(20).mean()*1.10).shift(1)
out=[]
for s,d in xs.items():
 d=d.copy(); d.date=pd.to_datetime(d.date); c=d.close.astype(float); vol=np.log(c/c.shift(1)).rolling(20).std(); raw=(np.log(c/c.shift(10))/vol).shift(1)
 out.append(pd.DataFrame({'date':d.date,'symbol':s,'raw':raw}))
x=pd.concat(out); x['median']=x.groupby('date').raw.transform('median'); x['vix_stress']=x.date.map(gate); x['signal']=np.where(x.vix_stress.fillna(False),-(x.raw-x['median']),x.raw-x['median']); x[['date','symbol','signal']].dropna().to_csv('scripts/miner_1_20330930_vix_relative_momentum_signal.csv',index=False)
print(len(x),x.signal.notna().mean())
