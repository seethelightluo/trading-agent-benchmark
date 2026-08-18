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
up=r.clip(lower=0).rolling(30).sum(); dn=(-r.clip(upper=0)).rolling(30).sum()
base=((up-dn)/(up+dn)).rolling(5).mean().shift(1)
ics=[]
for d in base.index:
 z=pd.concat([base.loc[d],(p.shift(-10)/p-1).loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append((d,z.iloc[:,0].corr(z.iloc[:,1])))
ic=pd.Series(dict(ics)).reindex(base.index)
prior=ic.shift(1).rolling(63,min_periods=32).mean()
adapt=base.mul(np.where(prior>=0,1.,-1.),axis=0)
fw=p.shift(-10)/p-1; rows=[]
for d in adapt.index:
 z=pd.concat([adapt.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',adapt.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1; rr=[]
 for d in adapt.index:
  z=pd.concat([adapt.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std())
print('prior sign fraction positive',float((prior>=0).mean()))
q.to_csv('scripts/miner_3_20340929_adaptive_tail63_signal.csv')
