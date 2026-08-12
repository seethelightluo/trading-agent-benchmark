import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; DEF=['XAU','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();d=r[DEF].mean(axis=1);risk=[x for x in U if x not in DEF]
for w in [3,5,15]:
 stress=(d.rolling(w).sum()-r[risk].mean(axis=1).rolling(w).sum())>0
 f=r.rolling(w).sum().sub(d.rolling(w).sum(),axis=0).div(r.rolling(20).std().replace(0,np.nan)).where(stress,np.nan); vals=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   c=z.f.corr(z.y)
   if np.isfinite(c):vals.append(c)
 q=pd.Series(vals);
 if w==3: f.to_csv('scripts/miner_3_20310918_defensive_lead_3d_signal.csv')
 print('w',w,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'coverage',round(f.notna().mean().mean(),4))
