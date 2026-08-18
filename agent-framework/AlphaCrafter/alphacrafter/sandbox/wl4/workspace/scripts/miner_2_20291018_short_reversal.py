import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): x=x.assign(date=pd.to_datetime(x.date)).set_index('date').sort_index();D[s]=x
c=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index();
# Multi-horizon reversal with volatility dampening; lag one day.
r=c.pct_change(); v=r.rolling(30,min_periods=20).std(); f=(-(c/c.shift(3)-1)/(v+1e-12)).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1;z=[];ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):z.append(q);ns.append(len(a))
 z=pd.Series(z);print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
 if h==10:
  for n in [250,500]:
   q=z.tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
