import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3200)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=np.log(P).diff()
# Cross-asset defensive breadth factor: reward assets with positive 20d relative strength,
# but emphasize relative strength when broad market breadth is weak; all inputs lagged.
cs20=r.rolling(20,min_periods=15).sum(); breadth=(cs20>0).mean(axis=1)
relative=cs20.sub(cs20.median(axis=1),axis=0)
F=(relative*(1+ (0.5-breadth).clip(-.5,.5))).shift(1)
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
D=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',D.index.min(),D.index.max(),'dates',len(D),'avgN',D.n.mean(),'coverage',D.n.mean()/15)
x=D.ic.dropna();print('daily IC',x.mean(),'std',x.std(),'ICIR',x.mean()/x.std(),'hit',(x>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028',None)]:
 y=D.loc[a:b].ic.dropna() if b else D.loc[a:].ic.dropna(); print(a,b,'dates',len(y),'IC',y.mean(),'ICIR',y.mean()/y.std() if len(y)>1 else np.nan)
for h in [1,3,5,10]:
 Y=r.rolling(h).sum().shift(-h+1) # return t+1 through t+h
 vals=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),Y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'))
 q=pd.Series(vals).dropna(); print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
rank=F.rank(axis=1,pct=True);print('turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
