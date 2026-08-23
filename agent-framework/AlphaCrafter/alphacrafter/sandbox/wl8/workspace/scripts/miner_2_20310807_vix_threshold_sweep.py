import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl=pd.DataFrame({s:get_stock_daily_data(s,3000).set_index('date')['close'].astype(float) for s in U}).sort_index(); v=get_index_daily_data('VIX',3000).set_index('date')['close'].astype(float).sort_index()
r=cl.pct_change(10); vol=cl.pct_change().rolling(20,min_periods=10).std()*np.sqrt(252); level=v.rolling(252,min_periods=60).rank(pct=True).reindex(cl.index)
def calc(th):
 g=((level-th)/(1-th)).clip(0,1); sig=(-r/vol.replace(0,np.nan)).mul(g,axis=0).shift(1); out=[]; ns=[]
 for d in cl.index:
  z=pd.concat([sig.loc[d],(cl.shift(-10)/cl-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):out.append(c);ns.append(len(z))
 x=np.array(out); return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
for t in [.70,.75,.80,.85,.90,.95]: print(t,calc(t))
