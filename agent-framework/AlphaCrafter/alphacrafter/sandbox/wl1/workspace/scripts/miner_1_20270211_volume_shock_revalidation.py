import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-10')
ds={}
for s in U:
 x=get_stock_daily_data(symbol=s,days=3000)
 if x is not None and len(x)>120: ds[s]=x.sort_values('date').drop_duplicates('date').set_index('date')
cal=sorted(set().union(*[set(x.index) for x in ds.values()]))
for h in [5,10,20]:
 rows=[]; ns=[]; dates=[]
 for d in cal:
  if d>cut: continue
  z=[]; y=[]
  for s,x in ds.items():
   if d not in x.index: continue
   k=x.index.get_loc(d); k=k.stop-1 if isinstance(k,slice) else k
   c=pd.to_numeric(x.close,errors='coerce'); v=pd.to_numeric(x.volume,errors='coerce').replace(0,np.nan)
   if k<65 or k+h>=len(x) or x.index[k+h]>cut: continue
   vr=np.log(v.iloc[k-4:k+1].mean()/v.iloc[k-59:k+1].mean()); sig=-(c.iloc[k]/c.iloc[k-5]-1)*vr; fr=c.iloc[k+h]/c.iloc[k]-1
   if np.isfinite(sig) and np.isfinite(fr): z.append(sig); y.append(fr)
  if len(z)>=8 and np.std(z)>0 and np.std(y)>0: rows.append(np.corrcoef(z,y)[0,1]);ns.append(len(z));dates.append(d)
 a=np.array(rows); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('recent_252',round(a[-252:].mean(),6),round(a[-252:].mean()/a[-252:].std(ddof=1),6),'last_date',dates[-1])
