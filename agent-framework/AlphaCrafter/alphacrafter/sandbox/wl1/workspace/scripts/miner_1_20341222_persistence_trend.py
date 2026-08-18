import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# interpretable persistence-adjusted trend: 40d return, scaled by volatility,
# multiplied by fraction of positive days over trailing 20d; lag one day.
ret=p.pct_change(40); vol=r.rolling(40).std()*np.sqrt(252); persist=(r>0).rolling(20).mean()
f=(ret/vol)*persist
f=f.shift(1)
rows=[]
for h in [5,10,20,40]:
 fr=p.shift(-h)/p-1
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [5,10,20,40]:
 q=o[o.h==h]; print('h',h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
 q=o[(o.h==10)&(o.date>=a)&(o.date<=b)]; print(a,b,'dates',len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
# signal turnover and latest
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20341222_persistence_trend_signal.csv')
q=o[o.h==10]; q.to_csv('scripts/miner_1_20341222_persistence_trend_ic.csv',index=False)
