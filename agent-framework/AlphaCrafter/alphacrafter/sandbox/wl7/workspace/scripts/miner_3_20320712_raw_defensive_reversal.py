import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; DEF=['XAU','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); dates=P.index
for window in [3,5,10]:
 out=[]; ns=[]
 for i,t in enumerate(dates):
  if i<window+2: continue
  x=(L-L.shift(window)).iloc[i]; f=-(x-x[DEF].median()); fw=L.shift(-1).iloc[i]-L.iloc[i]; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 z=pd.Series(out); print('window',window,'dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
