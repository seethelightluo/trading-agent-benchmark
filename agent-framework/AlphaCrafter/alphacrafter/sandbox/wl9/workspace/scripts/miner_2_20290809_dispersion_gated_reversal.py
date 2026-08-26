import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return pd.Series(d.close.values,index=pd.to_datetime(d.date)).astype(float).sort_index()
px=pd.DataFrame({s:g(s) for s in U}).sort_index(); print('shape',px.shape,'last',px.tail(1).notna().sum().sum())
r5=px.pct_change(5); disp=r5.std(axis=1); base=disp.rolling(120,min_periods=60).mean()/disp.rolling(120,min_periods=60).median(); sig=-r5*(1+0.5*base.clip(0,3).fillna(1))
for h in [5,10,20]:
 vals=[]; dates=[]; ns=[]; fwd=px.shift(-h)/px-1
 for d in sig.index:
  z=pd.concat([sig.loc[d].rename('s'),fwd.loc[d].rename('f')],axis=1).dropna()
  if len(z)>=8:
   q=z.s.rank().corr(z.f.rank())
   if np.isfinite(q): vals.append(q); dates.append(d); ns.append(len(z))
 a=np.array(vals); dates=np.array(dates,dtype='datetime64[ns]')
 def met(q): return (round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round(np.mean(q>0),4)) if len(q)>1 else ('nan','nan','nan')
 print('h',h,'dates',len(a),'mean_n',round(np.mean(ns),2) if ns else 0,'coverage',round(np.mean(ns)/15,4) if ns else 0,'IC/ICIR/hit',met(a))
 if h==10:
  for lab,cut in [('recent252','2028-08-01'),('2029','2029-01-01')]: print(lab,'n',sum(dates>=np.datetime64(cut)),'IC/ICIR/hit',met(a[dates>=np.datetime64(cut)]))