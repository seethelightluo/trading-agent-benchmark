import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,4000)
    if x is None: x=get_index_daily_data(s,4000)
    if x is not None:
        x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); ret=p.pct_change()
# One interpretable idea: cross-sectional relative acceleration, conditioned by
# breadth dispersion. High unanimity strengthens trend continuation; low breadth
# weakens it. All inputs are lagged before the forward-return test.
r20=p.pct_change(20); r60=p.pct_change(60)
rel20=r20.sub(r20.median(axis=1),axis=0); rel60=r60.sub(r60.median(axis=1),axis=0)
acc=rel20-rel60
bread=(r20>0).mean(axis=1)
# symmetric confidence: 1 at neutral breadth, 1.5 at extreme breadth
confidence=(1+2*(bread-0.5).abs()).clip(1,1.5)
sig=acc.mul(confidence,axis=0).shift(1)
rows=[]
for d in sig.index:
    z=pd.concat([sig.loc[d],p.shift(-10).loc[d]/p.loc[d]-1],axis=1).dropna()
    if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/len(U))
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 a=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],p.shift(-h).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(a))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std())
q.to_csv('scripts/miner_2_20350608_breadth_confidence_ic.csv')
sig.to_csv('scripts/miner_2_20350608_breadth_confidence_signal.csv',index_label='date')
