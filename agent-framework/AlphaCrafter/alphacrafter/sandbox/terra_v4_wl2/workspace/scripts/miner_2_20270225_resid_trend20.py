import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x): return x.set_index('date')['close']
  except: pass
px=pd.concat({s:get(s) for s in U},axis=1).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
# novel: medium-term trend residualized by contemporaneous cross-sectional market beta,
# and scaled by idiosyncratic volatility; avoids common risk and noisy raw momentum.
market=r.mean(axis=1); beta=r.rolling(60,min_periods=30).cov(market).div(market.rolling(60,min_periods=30).var(),axis=0)
res=r-beta.mul(market,axis=0); residvol=res.rolling(20,min_periods=15).std()
sig=(res.rolling(20,min_periods=20).sum()/residvol.replace(0,np.nan)).shift(1)
for h in [1,3,5,10]:
 y=px.shift(-h)/px-1; vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
# by year for 1d
for yr in sorted(set(sig.index.year)):
 vals=[]
 for d in sig.index[sig.index.year==yr]:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 if vals: print('year',yr,'n',len(vals),'IC',round(np.mean(vals),5),'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),5))
