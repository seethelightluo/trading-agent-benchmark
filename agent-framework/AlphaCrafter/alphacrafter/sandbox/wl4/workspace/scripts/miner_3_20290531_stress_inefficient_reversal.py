import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close.astype(float)
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
ret=p.pct_change(20); eff=ret.abs()/r.abs().rolling(20).sum(); vol=r.rolling(20).std()
base=(-ret*(1-eff)/vol)
csvol=vol.median(axis=1); baseline=csvol.rolling(60,min_periods=40).median(); stress=(csvol/baseline).clip(.5,2.0)
f=base.mul(1+0.5*(stress-1),axis=0).shift(1).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]; cov=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a)); cov.append(len(a)/len(U))
 z=pd.Series(vals).dropna(); print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f} coverage={np.mean(cov):.4f}')
 if h==10:
  for n in [250,500]:
   q=z.tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('coverage_panel',f.notna().mean().mean(),'rank_turn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('instruments',len(D),'range',p.index.min().date(),p.index.max().date())
