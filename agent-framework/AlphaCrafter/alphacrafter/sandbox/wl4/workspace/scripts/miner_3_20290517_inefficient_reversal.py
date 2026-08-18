import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close.astype(float)
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change();
# Contrarian return, amplified when the move is path-inefficient/choppy; lagged one day
ret=p.pct_change(20); efficiency=ret.abs()/r.abs().rolling(20).sum()
vol=r.rolling(20).std(); fac=(-ret*(1-efficiency)/vol).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; out=[]
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: out.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
 print(f'h={h} dates={len(q)} avgN={z.n.mean():.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
 if h==10:
  for n in [250,500]:
   q2=q.tail(n); print(f'recent{n} IC={q2.mean():.6f} ICIR={q2.mean()/q2.std(ddof=1):.6f}')
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
