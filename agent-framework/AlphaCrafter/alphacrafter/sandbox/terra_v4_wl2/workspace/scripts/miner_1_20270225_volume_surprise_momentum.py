import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(a):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(a,days=4000)
   if d is not None:return d
  except: pass
D={a:get(a) for a in U}; D={a:d for a,d in D.items() if d is not None}
C=pd.DataFrame({a:d.set_index('date').close.astype(float) for a,d in D.items()}).sort_index(); V=pd.DataFrame({a:d.set_index('date').volume.astype(float) for a,d in D.items()}).reindex(C.index)
mom=C/C.shift(20)-1; vr=V.rolling(20).mean()/V.rolling(60).mean()-1
# Volume-surprise-confirmed medium-term momentum, lagged one day.
f=(mom*(1+vr.clip(-.5,.5))).shift(1)
f.stack().rename('signal').to_csv('../persistent/factor_signals_miner_1_20270225_volume_surprise_momentum.csv')
print('assets',len(D),'dates',len(C),'rows',f.stack().size,'coverage',f.notna().mean().mean())
for h in [1,5,10]:
 y=C.shift(-h)/C-1; z=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 s=pd.Series(z).dropna();print('H',h,'n_dates',len(s),'avg_n',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',np.mean(s>0))
print('turnover',f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
