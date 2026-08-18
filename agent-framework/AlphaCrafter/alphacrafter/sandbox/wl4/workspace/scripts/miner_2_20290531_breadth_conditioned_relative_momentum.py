import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close.astype(float)
p=pd.concat(D,axis=1,sort=True).sort_index(); r=p.pct_change()
ret30=p.pct_change(30); breadth=(ret30>0).sum(axis=1)/ret30.notna().sum(axis=1)
market=r.mean(axis=1); resid30=ret30.sub(market.rolling(30).sum(),axis=0)
vol=r.rolling(30).std()*np.sqrt(30)
state=pd.Series(np.where(breadth>=0.5,1.0,-0.35),index=breadth.index)
fac=resid30.div(vol).mul(state,axis=0).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; out=[]
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: out.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
 print(f'h={h} dates={len(q)} avgN={z.n.mean():.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 if h==10:
  for n in [250,500,1000]:
   q2=q.tail(n); print(f'recent{n} IC={q2.mean():.6f} ICIR={q2.mean()/q2.std(ddof=1):.6f}')
print('assets',len(D),'coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
