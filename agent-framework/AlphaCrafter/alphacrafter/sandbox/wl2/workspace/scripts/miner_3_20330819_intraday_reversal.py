import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in [get_index_daily_data,get_stock_daily_data]:
  try:
   d=f(s,days=5000)
   if d is not None and len(d)>100:return d
  except: pass
xs={s:get(s) for s in U}; xs={s:d for s,d in xs.items() if d is not None}
def run(h):
 rr=[]
 for s,d in xs.items():
  c=d.close.astype(float);o=d.open.astype(float); hi=d.high.astype(float);lo=d.low.astype(float)
  v=np.log(c/c.shift(1)).rolling(20).std()
  # close location / intraday reversal, lagged
  f=(-(c-o)/c/v).clip(-4,4).shift(1); r=c.shift(-h)/c-1
  rr.append(pd.DataFrame({'date':d.date,'f':f,'r':r}).dropna())
 x=pd.concat(rr); z=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:z.append((dt,g.f.corr(g.r,method='spearman')))
 q=pd.Series(dict(z)).dropna();return q
for h in [1,3,5,10]:
 q=run(h);print(h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
