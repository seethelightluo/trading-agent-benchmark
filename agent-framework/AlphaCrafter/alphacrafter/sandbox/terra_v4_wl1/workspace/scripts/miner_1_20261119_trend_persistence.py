import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=2200)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date');D[s]['r']=D[s].close.pct_change()
all_dates=sorted(set().union(*[set(x.index) for x in D.values()]))
def calc(h):
 out=[]
 for dt in all_dates:
  fs=[]; rs=[]
  for x in D.values():
   if dt not in x.index: continue
   z=x.loc[:dt]; ix=x.index.get_loc(dt)
   if len(z)<65 or ix+h>=len(x): continue
   v20=z.r.iloc[-20:].std();v60=z.r.iloc[-60:].std()
   if not np.isfinite(v20) or v20<=0: continue
   f=(z.close.iloc[-1]/z.close.iloc[-6]-1)/(v20*np.sqrt(5))-.25*v60*np.sqrt(252)
   fs.append(f);rs.append(x.close.iloc[ix+h]/x.close.iloc[ix]-1)
  if len(fs)>=8: out.append((dt,len(fs),np.corrcoef(fs,rs)[0,1]))
 return pd.DataFrame(out,columns=['date','n','ic']).dropna()
for h in [1,5,10]:
 r=calc(h); q=r.ic
 print(h,'dates',len(r),'avg_names',r.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'coverage',r.n.mean()/15)
 if h==1: print('period',r.date.min(),r.date.max())
