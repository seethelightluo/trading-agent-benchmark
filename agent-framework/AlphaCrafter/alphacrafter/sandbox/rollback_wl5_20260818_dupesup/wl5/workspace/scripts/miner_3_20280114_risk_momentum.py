import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=2600).set_index('date')['close'].astype(float) for s in U}
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); results={}
for look in [10,20,40]:
 for h in [5,10]:
  q=[]
  for i in range(max(look,25),len(P)-h):
   v={}; f={}
   for s in P.columns:
    vol=R[s].iloc[i-19:i+1].std(); ret=P[s].iloc[i]/P[s].iloc[i-look]-1
    if np.isfinite(vol) and vol>0 and pd.notna(ret) and pd.notna(P[s].iloc[i+h]): v[s]=ret/vol; f[s]=P[s].iloc[i+h]/P[s].iloc[i]-1
   if len(v)>=8:q.append(pd.Series(v).corr(pd.Series(f).reindex(v)))
  z=pd.Series(q).dropna(); results[(look,h)]=(len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
  print('look',look,'h',h,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
