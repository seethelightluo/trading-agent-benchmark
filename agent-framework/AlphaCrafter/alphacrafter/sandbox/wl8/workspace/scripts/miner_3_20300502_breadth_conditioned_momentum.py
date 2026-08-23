import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:get_stock_daily_data(s,5000).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change()
r20=P/P.shift(20)-1; r60=P/P.shift(60)-1
breadth=(r20>0).mean(axis=1); regime=np.where(breadth>=.5,1,-1)
# in broad positive regimes follow 20d trend; in negative regimes prefer relative rebound, with 60d confirmation
F=r20.mul(regime,axis=0) - .25*r60
F=F.sub(F.median(axis=1),axis=0).clip(-8,8)
res=[]
for i in range(len(P)-10):
 z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: res.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(res,columns=['date','n','ic']).set_index('date'); q=x.ic
print('dates',len(x),'range',x.index.min(),x.index.max(),'avg_n',x.n.mean(),'coverage',x.n.mean()/15)
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,20]:
 a=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:a.append(z.f.corr(z.y,method='spearman'))
 print(h,np.nanmean(a),len(a))
for n in [180,360]: print('recent',n,q.tail(n).mean(),q.tail(n).mean()/q.tail(n).std(ddof=1))
print('last',P.index.max())
