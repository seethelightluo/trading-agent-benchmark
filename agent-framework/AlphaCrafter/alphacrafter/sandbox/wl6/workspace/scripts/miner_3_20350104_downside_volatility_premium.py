import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index()
r=np.log(P/P.shift(1)); down=r.where(r<0,0.0)
# lagged blended downside realized volatility; downside risk premium tests whether stressed assets subsequently rebound
f=((down.rolling(20).std()*np.sqrt(20))+(down.rolling(60).std()*np.sqrt(60)))/2
f=f.shift(1)
rows=[]
for h in [5,10,20,40]:
 fw=P.shift(-h)/P-1; ic=[]; ns=[]; turnover=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
   sig=z.iloc[:,0].rank(pct=True)
   if prev is not None: turnover.append(np.mean(abs(sig-prev)))
   prev=sig
 a=np.asarray(ic); a=a[np.isfinite(a)]
 print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(U),4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),3),'hit',round(np.mean(a>0),4),'turn',round(np.mean(turnover),4))
out=[]
for dt in f.index:
 for s in f:
  if pd.notna(f.loc[dt,s]):out.append([str(dt.date()),s,float(f.loc[dt,s])])
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20350104_downside_volatility_premium_signal.csv',index=False)
