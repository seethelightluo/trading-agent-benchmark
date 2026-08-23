import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,3000); x=x.sort_values('date').set_index('date'); r=x.close.pct_change(); sig=-(x.open/x.close.shift(1)-1)/r.rolling(20).std(); D[s]=(sig,x.close)
for h in [1,3,5,10,20]:
 vals=[]
 for dt in sorted(set().union(*[set(v[0].index) for v in D.values()])):
  z=[]
  for s,(sig,p) in D.items():
   if dt in sig.index:
    i=p.index.get_loc(dt)
    if i+h<len(p) and np.isfinite(sig.loc[dt]): z.append((sig.loc[dt],p.iloc[i+h]/p.iloc[i]-1))
  if len(z)>=8:
   q=pd.DataFrame(z,columns=['s','f']); c=q.s.corr(q.f,method='spearman')
   if np.isfinite(c): vals.append(c)
 a=np.array(vals); print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
