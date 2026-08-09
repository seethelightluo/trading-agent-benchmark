import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=2200)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);x=x.sort_values('date').drop_duplicates('date').set_index('date');x['r']=x.close.pct_change();D[s]=x
for h in [1,5,10]:
 rows=[]
 dates=sorted(set().union(*[set(x.index) for x in D.values()]))
 for dt in dates:
  fs=[];rs=[]
  for x in D.values():
   if dt not in x.index: continue
   z=x.loc[:dt];ix=x.index.get_loc(dt)
   if len(z)<65 or ix+h>=len(x):continue
   v=z.r.iloc[-20:].std(); v60=z.r.iloc[-60:].std()
   if not np.isfinite(v) or v<=0:continue
   fs.append(-v/(v60 if v60>0 else v));rs.append(x.close.iloc[ix+h]/x.close.iloc[ix]-1)
  if len(fs)>=8:rows.append(np.corrcoef(fs,rs)[0,1])
 q=pd.Series(rows).dropna();print(h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
