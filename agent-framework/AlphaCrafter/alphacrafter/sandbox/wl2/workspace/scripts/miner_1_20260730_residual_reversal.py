import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
f={}
for s in S:
 d=get_stock_daily_data(s,days=1800)
 if d is not None:
  x=d[['date','close']].drop_duplicates('date'); f[s]=x.set_index('date').close
p=pd.DataFrame(f).sort_index(); ret=p.pct_change(); m=ret.mean(axis=1)
# residual 20d return: asset cumulative return minus market cumulative return; contrarian residual
res=p.pct_change(20).sub(p.pct_change(20).mean(axis=1),axis=0)
sig=-res
rows=[]
for dt in sig.index:
 v=pd.concat([sig.loc[dt],p.pct_change().shift(-1).loc[dt]],axis=1).dropna()
 if len(v)>=8 and v.iloc[:,0].nunique()>1: rows.append((dt,v.iloc[:,0].corr(v.iloc[:,1]),len(v)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.3f turnover %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean(),sig.rank(pct=True).diff().abs().stack().mean()))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-29')]:
 z=r.loc[a:b];print(a,len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
for h in [5,10]:
 y=p.pct_change(h).shift(-h); q=[]
 for dt in sig.index:
  v=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(v)>=8:q.append(v.iloc[:,0].corr(v.iloc[:,1]))
 print('h',h,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
