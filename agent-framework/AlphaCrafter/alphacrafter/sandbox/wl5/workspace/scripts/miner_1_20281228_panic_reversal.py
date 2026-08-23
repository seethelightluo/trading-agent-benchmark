import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
r5=p.pct_change(5); high=(vix>vix.rolling(60,min_periods=30).median()).astype(float)
sig=r5.mul(-high,axis=0).sub(r5.mul(-high,axis=0).mean(axis=1),axis=0)
for h in [5,10,20]:
 fwd=p.shift(-h)/p-1;ics=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):ics.append(q);ns.append(len(z))
 a=np.array(ics); recent=a[-252:]
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recentIC',round(recent.mean(),6),'recentIR',round(recent.mean()/recent.std(ddof=1),6))
sig.index.name='date';sig.to_csv('scripts/miner_1_20281228_panic_reversal_signal.csv')
print('coverage',round(sig.notna().sum().sum()/(len(sig)*15),4))
