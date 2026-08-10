import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change();
# Stress-conditioned continuation: after lagged broad downside breadth, favor 5d winners.
b=(r.lt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1)
for t in [.50,.55,.60,.65]:
 f= r.rolling(5).sum().where(b>=t); f=f.sub(f.median(axis=1),axis=0)
 for h in [1,5]:
  fr=px.shift(-h)/px-1;a=[];ns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  a=np.asarray(a); print(t,h,len(a),round(np.mean(ns),2),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),round(np.mean(a>0),4))
# artifact for best candidate is generated only for inspection
f=r.rolling(5).sum().where(b>=.50); f=f.sub(f.median(axis=1),axis=0)
out=f.stack().rename('signal').rename_axis(['date','symbol']).to_frame();out.to_csv('../persistent/factor_signals_miner_3_20270225_stress_continuation.csv')
print('coverage',f.notna().sum().sum()/f.size,'dates',len(f),'assets',len(U))
