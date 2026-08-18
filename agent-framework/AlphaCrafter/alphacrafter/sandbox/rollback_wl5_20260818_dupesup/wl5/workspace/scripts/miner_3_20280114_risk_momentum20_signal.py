import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:get_stock_daily_data(s,days=2600).set_index('date')['close'].astype(float) for s in U}).sort_index().ffill(); R=P.pct_change(); out=[]
for i in range(25,len(P)-10):
 v={}
 for s in P.columns:
  vol=R[s].iloc[i-19:i+1].std(); ret=P[s].iloc[i]/P[s].iloc[i-20]-1
  if np.isfinite(vol) and vol>0 and pd.notna(ret) and pd.notna(P[s].iloc[i+10]): v[s]=ret/vol
 if len(v)>=8:
  for s,x in v.items():out.append({'date':P.index[i],'symbol':s,'signal':x,'forward_return_10d':P[s].iloc[i+10]/P[s].iloc[i]-1})
pd.DataFrame(out).to_csv('scripts/miner_3_20280114_risk_momentum20_signal.csv',index=False)
