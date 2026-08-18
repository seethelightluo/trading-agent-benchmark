import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None: x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Candidate: lagged 40-day upside/downside asymmetry, smoothed, with a slower
# 120-observation sign filter learned only from prior daily cross-sectional IC.
up=r.clip(lower=0).rolling(40).sum(); dn=(-r.clip(upper=0)).rolling(40).sum()
base=((up-dn)/(up+dn)).rolling(5).mean().shift(1)
fw10=p.shift(-10)/p-1
raw=[]
for d in base.index:
 z=pd.concat([base.loc[d],fw10.loc[d]],axis=1).dropna()
 if len(z)>=8: raw.append((d,z.iloc[:,0].corr(z.iloc[:,1])))
ic=pd.Series(dict(raw)).reindex(base.index)
prior=ic.shift(1).rolling(120,min_periods=60).mean()
sig=base.mul(np.where(prior>=0,1.,-1.),axis=0)
rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fw10.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1; rr=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std())
print('prior positive',float((prior>=0).mean()))
q.to_csv('scripts/miner_3_20341027_adaptive_tail120_signal.csv')
