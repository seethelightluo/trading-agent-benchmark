import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
EQ=U[:8]
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None:return d
  except: pass
D={s:g(s) for s in U}; C=pd.DataFrame({s:d.set_index('date').close.astype(float) for s,d in D.items() if d is not None}).sort_index(); R=C.pct_change()
# Cross-asset leadership: reverse 5d return only when equity breadth is weak; center by all assets.
b=(R[EQ].gt(0).mean(axis=1)<.375).shift(1); raw=-R.rolling(5).sum(); f=raw.where(b); f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'avg_n',np.mean(ns) if ns else 0,'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
print('active',int(b.sum()),'coverage',f.notna().mean().mean(),'datecov',f.notna().any(axis=1).mean())
f.stack().rename('signal').reset_index().to_csv('../persistent/factor_signals_miner_3_20270225_breadth_reversal5_new.csv',index=False)
