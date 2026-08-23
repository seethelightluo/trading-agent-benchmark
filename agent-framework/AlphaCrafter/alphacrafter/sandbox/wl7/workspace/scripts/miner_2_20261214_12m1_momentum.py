import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=get_account_dict()['watch_list']; cutoff=pd.Timestamp('2026-12-13'); ds={}
for s in U:
 try: ds[s]=get_stock_daily_data(s,2600)
 except Exception: ds[s]=get_index_daily_data(s,2600)
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in ds.items()}).sort_index().loc[:cutoff]
r=cl.pct_change(); sig=(cl.shift(20)/cl.shift(260)-1)/(r.shift(1).rolling(60).std()*np.sqrt(60)); fwd=cl.shift(-1)/cl-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('cutoff',cutoff.date(),'dates',len(df),'avgN',df.n.mean(),'coverage',sig.notna().sum(axis=1).mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(df.ic.mean(),df.ic.mean()/df.ic.std(),(df.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=df.loc[a:b].ic; print(a,b,len(q),'ic',q.mean(),'icir',q.mean()/q.std())
for h in [5,10,20]:
 y=cl.shift(-h)/cl-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'dates',len(rr),'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/np.nanstd(rr))
