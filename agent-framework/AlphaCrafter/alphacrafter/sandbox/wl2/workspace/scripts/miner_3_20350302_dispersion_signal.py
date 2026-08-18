import numpy as np, pandas as pd, json
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_stock_daily_data(s,days=5000)
 except:d=None
 if d is None or len(d)<150:
  try:d=get_index_daily_data(s,days=5000)
  except:d=None
 return d
xs={s:get(s) for s in U}; px=pd.DataFrame({s:d.set_index('date').close for s,d in xs.items() if d is not None}).sort_index(); r=px.pct_change(); disp=r.rolling(20).std().mean(axis=1); gate=disp.shift(1)>disp.shift(1).rolling(120).median(); sig=-(px.pct_change(10).shift(1))/r.rolling(30).std().shift(1); sig=sig.where(gate, np.nan)
out=sig.reset_index().rename(columns={'index':'date'}); out.to_csv('../persistent/miner_3_20350302_dispersion_reversal_signal.csv',index=False)
print('wrote',len(out),out['date'].min(),out['date'].max())
PY='20350302'
