import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').sort_index()
c=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index();r=c.pct_change();v=r.rolling(20,min_periods=15).std(); shock=(v/v.rolling(60,min_periods=30).median()).clip(.5,3)
# Reversal signal: fade recent moves, with extra weight when volatility has recently shocked higher.
f=(-c.pct_change(5).div(v))*shock
f=f.sub(f.mean(axis=1),axis=0).shift(1).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'range',c.index.min().date(),c.index.max().date())
for h in [1,5,10,20]:
 z=[];ns=[];cov=[];fw=c.shift(-h)/c-1
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):z.append(q);ns.append(len(a));cov.append(len(a)/15)
 z=pd.Series(z);print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f} coverage={np.mean(cov):.4f}')
 if h==10:
  for n in [250,500]:
   q=z.tail(n);print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('panel_coverage',f.notna().mean().mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
