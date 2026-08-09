import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); fac=p.pct_change(20)*(2*r.rolling(20).apply(lambda x:np.mean(x>0),raw=True)-1)/(r.rolling(20).std()+1e-8)
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270325_trend_consistency_signal.csv',index=False); print(len(out),out.date.min(),out.date.max(),out.signal.notna().mean())
