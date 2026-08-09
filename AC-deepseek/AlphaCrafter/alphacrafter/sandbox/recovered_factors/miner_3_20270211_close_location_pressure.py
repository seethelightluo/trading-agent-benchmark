import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; F={}; R={}
for s in A:
 d=get_stock_daily_data(s,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 c=pd.to_numeric(d.close,errors='coerce'); h=pd.to_numeric(d.high,errors='coerce'); l=pd.to_numeric(d.low,errors='coerce')
 cl=(2*c-h-l)/(h-l).replace(0,np.nan); F[s]=cl.rolling(15,min_periods=10).mean(); R[s]=c.pct_change()
fac=pd.concat(F,axis=1); ret=pd.concat(R,axis=1)
print('FACTOR close_location_pressure_15; dates',len(fac),'assets',len(A))
def ev(x,h):
 fw=(1+ret).rolling(h,min_periods=h).apply(np.prod,raw=True).shift(-h)
 vals=[]; ns=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals); return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),np.mean(ns)
for h in [1,5,10,20]: print('h',h,ev(fac,h))
for nm,mask in [('2020',fac.index.year==2020),('2021_22',(fac.index.year>=2021)&(fac.index.year<=2022)),('2023_24',(fac.index.year>=2023)&(fac.index.year<=2024)),('2025_26',fac.index.year>=2025)]: print('REGIME',nm,ev(fac.loc[mask],1))
print('coverage',fac.notna().stack().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for fid in ['ravmom','trend','shortrev','vol']:
 E={}
 for s in A:
  r=R[s]
  if fid=='ravmom': E[s]=r.rolling(20).sum()
  elif fid=='trend': E[s]=r.rolling(20).sum()/r.rolling(20).std()
  elif fid=='shortrev': E[s]=-r.rolling(5).sum()/r.rolling(10).std()
  else: E[s]=-r.rolling(20).std()
 e=pd.concat(E,axis=1); z=pd.concat([fac.stack(),e.stack()],axis=1).dropna(); print('LIBCORR',fid,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),'cells',len(z))
