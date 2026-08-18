import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill(); ret=px.pct_change()
mom=px.pct_change(40); persistence=(ret.gt(0).rolling(40).mean()-0.5)*2
vol=ret.rolling(60).std()*np.sqrt(252); factor=(mom*persistence)/vol.replace(0,np.nan); factor=factor.shift(1)
out=[]
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).dropna(); sd=q.ic.std(ddof=1); mean=q.ic.mean()
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(mean,6),'ICIR',round(mean/sd,6),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.sum()/(len(q)*len(U)),4))
 for label,sub in [('recent260',q.tail(260)),('recent520',q.tail(520)),('2029_2031',q[(q.date>='2029-01-01')&(q.date<'2032-01-01')]),('2032_2033',q[q.date>='2032-01-01'])]:
  if len(sub)>2: print(' ',label,len(sub),round(sub.ic.mean(),6),round(sub.ic.mean()/sub.ic.std(ddof=1),6))
r=factor.rank(axis=1,pct=True); print('TURNOVER',round(float(r.diff().abs().mean(axis=1).dropna().mean()),6),'DATES',px.index.min(),px.index.max(),'N',len(frames),'rows',len(px))
factor.index.name='date'; factor.reset_index().to_csv('scripts/artifacts/miner_2_20330915_persistence_quality_signal.csv',index=False)
