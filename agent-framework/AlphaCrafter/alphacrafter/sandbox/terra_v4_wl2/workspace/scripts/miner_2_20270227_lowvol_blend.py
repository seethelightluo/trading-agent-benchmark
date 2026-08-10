import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# defensive low-volatility signal, with lagged 20d realized vol and 5d trend blend
vol=r.rolling(20,min_periods=15).std(); mom=r.rolling(5).sum()
for w in [0,0.25,0.5,0.75]:
 sig=(-vol + w*mom).shift(1)
 for h in [1,5,10]:
  fr=px.shift(-h)/px-1; vals=[]; ns=[]
  for d in sig.index:
   z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  a=np.array(vals); print('lowvolblend',w,'h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
