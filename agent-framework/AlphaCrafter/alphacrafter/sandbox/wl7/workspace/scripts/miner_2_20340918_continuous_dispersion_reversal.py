import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5200)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float).sort_index()
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); bench=r.mean(axis=1); disp=r.sub(bench,axis=0).std(axis=1)
# Continuous, winsorized dispersion weight avoids binary coverage loss.
rel=-(p.pct_change(5)).sub(p.pct_change(5).mean(axis=1),axis=0)
vol=r.rolling(20).std()*np.sqrt(20)
base=rel/vol
med=disp.rolling(60).median(); scale=(disp/med).clip(0.5,2.0)
sig=base*scale.values[:,None]
rows=[]
for d in sig.index:
 fut=p.pct_change(10).shift(-10).loc[d]
 z=pd.concat([sig.loc[d],fut],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,q in [('full',x),('recent180',x.tail(180)),('recent500',x.tail(500)),('recent1000',x.tail(1000))]:
 print(name,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
for h in [1,5,20]:
 rr=p.pct_change(h).shift(-h); vals=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],rr.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('DATA',len(p),p.index.min(),p.index.max(),'assets',len(D),'coverage',sig.notna().sum(axis=1).mean()/len(U),'turnover',sig.rank(pct=True,axis=1).diff().abs().sum(axis=1).div(2*len(U)).mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340918_continuous_dispersion_reversal_signal.csv',index=False)
