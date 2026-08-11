import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ds={}
for s in U:
 x=get_stock_daily_data(symbol=s,days=3000)
 if x is not None and len(x)>120: ds[s]=x.sort_values('date').set_index('date')
cal=ds['SPX'].index
for horizon in [5,10,20]:
 rows=[]
 for d in cal:
  z=[]; f=[]
  for s,x in ds.items():
   if d not in x.index: continue
   k=x.index.get_loc(d); c=pd.to_numeric(x.close,errors='coerce');v=pd.to_numeric(x.volume,errors='coerce').replace(0,np.nan)
   if isinstance(k,slice): k=k.stop-1
   if k<65 or k+horizon>=len(x): continue
   vr=np.log(v.iloc[k-4:k+1].mean()/v.iloc[k-59:k+1].mean())
   sig=-(c.iloc[k]/c.iloc[k-5]-1)*vr
   fr=c.iloc[k+horizon]/c.iloc[k]-1
   if np.isfinite(sig) and np.isfinite(fr):z.append(sig);f.append(fr)
  if len(z)>=8 and np.std(z)>0 and np.std(f)>0: rows.append(np.corrcoef(z,f)[0,1])
 q=np.array(rows); print(horizon,len(q),q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0))
