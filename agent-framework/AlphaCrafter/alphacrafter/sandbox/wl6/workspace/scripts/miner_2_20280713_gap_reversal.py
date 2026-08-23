import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=4000)
   if x is not None:return x
  except:pass
S={}
for s in U:
 d=get(s)
 if d is None:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 # gap proxy: open relative to prior close, lagged one completed session; normalize by 20d vol
 gap=d.open/d.close.shift(1)-1
 vol=d.close.pct_change().rolling(20).std()
 sig=(-gap/vol).replace([np.inf,-np.inf],np.nan)
 S[s]=pd.DataFrame({'sig':sig,'f1':d.close.pct_change().shift(-1),'f5':d.close.shift(-5)/d.close-1,'f10':d.close.shift(-10)/d.close-1})
for h in ['f1','f5','f10']:
 rows=[]
 for dt in sorted(set().union(*[x.index for x in S.values()])):
  a=[(x.loc[dt].sig,x.loc[dt,h]) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt,h])]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','r']);rows.append(z.s.rank().corr(z.r.rank()))
 q=pd.Series(rows).dropna(); print(h,'dates',len(q),'avgN',sum(len([x for x in S.values() if dt in x.index]) for dt in []) ,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
