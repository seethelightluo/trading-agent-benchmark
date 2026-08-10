import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); d=g('DXY').set_index('date')['close'].reindex(px.index).ffill(); dr=d.pct_change()
# Dollar trend-conditioned relative momentum: reverse short momentum when dollar trend is strong, otherwise retain it.
dz=(dr.rolling(20).sum()/dr.rolling(60).std()).shift(1); sign=np.where(dz>0,-1,1); f=(r.rolling(5).sum().mul(sign,axis=0)); f=f.sub(f.median(axis=1),axis=0).shift(1)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
# artifact for admission horizon 5
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_dxy_conditioned_momentum.csv',index=False)
