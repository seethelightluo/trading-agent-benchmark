import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; O={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); O[s]=d.set_index('date')
P=pd.DataFrame({s:d.close for s,d in O.items()}).sort_index().ffill(); R=P.pct_change()
# close-to-open shock reversal, smoothed 3 sessions, normalized by prior 20d close volatility
shock=pd.DataFrame({s:(d.open/d.close.shift(1)-1) for s,d in O.items()}).reindex(P.index).ffill()
vol=R.rolling(20,min_periods=15).std().shift(1)
F=-shock.rolling(3).mean()/vol
for h in [1,5,10,20]:
 a=[]; cov=[]
 for i in range(25,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));cov.append(len(z)/15)
 a=pd.Series(a).dropna();print(h,len(a),a.mean(),a.mean()/a.std(),(a>0).mean(),np.mean(cov))
F.index.name='date';F.to_csv('scripts/miner_3_20350831_intraday_shock_signal.csv')
