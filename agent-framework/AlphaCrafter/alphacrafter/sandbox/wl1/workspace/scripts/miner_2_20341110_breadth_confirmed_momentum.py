import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=6000)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); mom=p/p.shift(20)-1
breadth=(r>0).rolling(10).mean().mean(axis=1)
f=mom.mul((breadth-0.5).clip(-0.5,0.5),axis=0).shift(1)
rows=[]
for d in f.index:
 j=p.index.searchsorted(d)
 if j+10>=len(p): continue
 z=pd.concat([f.loc[d],p.iloc[j+10]/p.iloc[j]-1],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('cutoff',x.index.max().date(),'dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.3f turnover %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean(),f.rank(axis=1).diff().abs().sum(axis=1).div(15).mean()))
for a,b in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-12-31')]:
 q=x.loc[a:b]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20,40]:
 rr=[]
 for d in f.index:
  j=p.index.searchsorted(d)
  if j+h>=len(p): continue
  z=pd.concat([f.loc[d],p.iloc[j+h]/p.iloc[j]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
f.to_csv('scripts/miner_2_20341110_breadth_confirmed_momentum_signal.csv')
