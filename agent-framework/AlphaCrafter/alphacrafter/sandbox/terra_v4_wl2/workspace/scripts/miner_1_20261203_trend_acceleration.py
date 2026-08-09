import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change()
# Trend acceleration: intermediate 20d return relative to slow 60d return, with volatility normalization.
F={'accel':R.rolling(20,min_periods=15).sum()-R.rolling(60,min_periods=45).sum(),
   'accel_norm':(R.rolling(20,min_periods=15).sum()-R.rolling(60,min_periods=45).sum())/R.rolling(60,min_periods=45).std(),
   'slow_sharpe':R.rolling(60,min_periods=45).sum()/R.rolling(60,min_periods=45).std()}
for n,f in F.items():
 for h in [1,5,10]:
  a=[]; ds=[]; ns=[]
  for i in range(len(P)-h):
   z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(P.index[i]);ns.append(len(z))
  a=np.array(a); print(n,h,'dates',len(a),'names',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
 print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),4))
 # yearly means daily
 for y in range(2020,2027):
  x=[a[j] for j,d in enumerate(ds) if d.year==y]
  if x: print(y,round(np.mean(x),4),round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else 0,len(x))
