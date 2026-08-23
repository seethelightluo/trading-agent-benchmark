import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_stock_daily_data(s,days=3600)
 except:continue
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change(); f=p.pct_change(10)/(r.rolling(30).std()*np.sqrt(30)+1e-8);f=f.clip(-5,5)
for h in [1,5,10]:
 fr=p.pct_change(h).shift(-h);v=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(z))
 ic=pd.Series(v,index=ds).dropna(); print('h',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round(np.mean(ic>0),4),'turn',round(f.diff().abs().mean().mean(),4))
 for n,a,b in [('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31')]:
  q=ic.loc[a:b];print(n,round(q.mean(),6) if len(q) else 'NA',len(q))
