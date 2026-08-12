import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Short-vs-medium reversal: fade 5d move relative to 20d trend, volatility scaled
f=-(r.rolling(5).sum()-0.25*r.rolling(20).sum())/r.rolling(20).std()
for h in [1,3,5,10]:
 a=[]; ns=[]
 for i in range(40,len(p)-h):
  z=pd.concat([f.iloc[i-1],r.rolling(h).sum().iloc[i+h-1]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
f.to_csv('scripts/miner_1_20310403_short_medium_reversal_signal.csv',index_label='date')
