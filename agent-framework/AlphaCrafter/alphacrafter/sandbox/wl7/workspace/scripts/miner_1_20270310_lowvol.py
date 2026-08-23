import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,2200)
 if x is None or len(x)<100:x=get_index_daily_data(s,2200)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); f=-r.rolling(20).std().shift(1); q=[]
for d in P.index:
 z=pd.concat([f.loc[d],r.shift(-1).loc[d]],axis=1).dropna()
 if len(z)>=8:q.append([d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)])
q=pd.DataFrame(q,columns=['date','ic','n']).set_index('date');print('dates',len(q),'avg_n',q.n.mean(),'assets',len(D),'coverage',q.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [5,10,20]:
 a=[]
 for d in P.index:
  z=pd.concat([f.loc[d],P.pct_change(h).shift(-h).loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
