import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): P[s]=d.set_index('date').close.astype(float)
pd0=pd.DataFrame(P).sort_index().ffill(); r=np.log(pd0).diff()
# Cross-asset beta-neutral short reversal: remove each asset's rolling 60d beta to equal-weight benchmark, then fade the 5d residual shock.
b=r.mean(axis=1); cov=r.rolling(60).cov(b); var=b.rolling(60).var()+1e-12
beta=cov.div(var,axis=0); resid=r-beta.mul(b,axis=0)
f=(-resid.rolling(5).sum()/ (resid.rolling(40).std()*np.sqrt(5)+1e-12)).shift(1)
rows=[]
for h in [1,3,5,10,20]:
 for i,t in enumerate(pd0.index[:-h]):
  z=pd.concat([f.loc[t],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((t,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [1,3,5,10,20]:
 q=x[x.h==h]; print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
for a,bn in [('2020-22',('2020','2022')),('2023-25',('2023','2025')),('2026-28',('2026','2028')),('2029+',('2029','2032'))]:
 q=x[(x.h==1)&(x.date.astype(str)>=bn[0])&(x.date.astype(str)<=bn[1])];print(a,'n',len(q),'IC',round(q.ic.mean(),6) if len(q) else None)
print('coverage',round(f.notna().stack().mean(),4),'dates',pd0.index.min(),pd0.index.max(),'assets',len(P))
