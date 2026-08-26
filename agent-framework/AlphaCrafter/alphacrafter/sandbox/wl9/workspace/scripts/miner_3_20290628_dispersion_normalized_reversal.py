import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.DataFrame({s:get(s) for s in U}).sort_index(); r5=px.pct_change(5)
med=r5.median(axis=1); disp=r5.sub(med,axis=0).abs().median(axis=1)
sig=-r5.sub(med,axis=0)/(disp+1e-6); fwd=px.shift(-10)/px-1
rows=[]; dates=[]; ns=[]
for d in sig.index:
 q=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(q)>=8:
  v=q.iloc[:,0].rank().corr(q.iloc[:,1].rank())
  if np.isfinite(v): rows.append(v); dates.append(d); ns.append(len(q))
a=np.array(rows,dtype=float); dates=np.array(dates,dtype='datetime64[ns]')
print('factor=dispersion_normalized_relative_reversal horizon=10')
print('data_dates',len(sig),'valid_dates',len(a),'mean_n',np.mean(ns) if ns else 0,'mean_coverage',np.mean(ns)/15 if ns else 0)
if len(a)==0: raise SystemExit('no valid cross-sectional observations')
print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for label,cut in [('online','2026-07-16'),('2028','2028-01-01'),('2029','2029-01-01'),('recent252','2028-06-28')]:
 z=a[dates>=np.datetime64(cut)]; print(label,'n',len(z),'IC %.8f ICIR %.8f hit %.5f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)) if len(z)>1 else label+' insufficient')
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; vals=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].rank().corr(q.iloc[:,1].rank())
   if np.isfinite(v):vals.append(v)
 z=np.array(vals); print('decay',h,'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
