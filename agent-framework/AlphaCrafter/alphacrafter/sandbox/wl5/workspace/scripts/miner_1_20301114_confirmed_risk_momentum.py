import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(frames).sort_index(); ret=px.pct_change(); mom=px.pct_change(30); vol=ret.rolling(40,min_periods=30).std()*np.sqrt(252); cons=(ret.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
sig=mom.div(vol.replace(0,np.nan))*cons; sig=sig.sub(sig.median(axis=1),axis=0)
out=sig.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}); out.to_csv('scripts/miner_1_20301114_confirmed_risk_momentum_signal.csv',index=False)
for h in [5,10,20]:
 fwd=px.shift(-h).div(px)-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).dropna(); ic=a.ic.mean(); sd=a.ic.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 print(h,'dates',len(a),'mean_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(icir,6),'hit',round((a.ic>0).mean(),4))
 if h==10:
  for name,lo,hi in [('2020-24','2020-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31'),('2028-29','2028-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   q=a[(a.date>=lo)&(a.date<=hi)]; print(name,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),6) if len(q)>1 else None)
print('turnover_proxy',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6),'dates',len(sig),'assets',len(px.columns)); print('range',px.index.min(),px.index.max())
