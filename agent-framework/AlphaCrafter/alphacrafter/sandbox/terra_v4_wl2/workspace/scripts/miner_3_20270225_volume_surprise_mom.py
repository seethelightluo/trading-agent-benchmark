import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=4000) for a in U}; C=pd.DataFrame({a:d.set_index('date').close.astype(float) for a,d in D.items() if d is not None}).sort_index(); V=pd.DataFrame({a:d.set_index('date').volume.astype(float) for a,d in D.items() if d is not None}).reindex(C.index)
R=C.pct_change(); mom=C/C.shift(20)-1; vr=V.rolling(20).mean()/V.rolling(60).mean()-1
# signed momentum weighted by volume surprise, with robust cross-sectional clipping
f=mom*(1+vr.clip(-.5,.5)); fr=C.shift(-1)/C-1
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 s=pd.Series(z).dropna();print(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),np.mean(s>0))
print('cov',f.notna().mean().mean(),'turn',f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').to_csv('../persistent/factor_signals_miner_3_20270225_volume_surprise_mom.csv')
