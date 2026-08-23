import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_stock_daily_data(s,days=3600)
 except:continue
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change();
# breakout distance, normalized by ATR-like vol, conditioned on positive slope
f=(p/p.rolling(20).max()-1)/(r.rolling(20).std()+1e-8)
f=f.clip(-5,5)
for h in [1,5,10]:
 fr=p.pct_change(h).shift(-h);v=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(z))
 ic=pd.Series(v,index=ds).dropna();print('h',h,'dates',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',np.mean(ic>0),'cov',np.mean(ns)/15)
 for n,a,b in [('25-26','2025-01-01','2026-12-31'),('27-28','2027-01-01','2028-12-31'),('29','2029-01-01','2029-12-31')]:
  q=ic.loc[a:b];print(n,q.mean() if len(q) else 'NA',len(q))
