import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-05-27')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index()
# lagged risk-adjusted 10d momentum: return ending t-1 / vol ending t-1
r=np.log(p).diff()
fac=(np.log(p).diff(10).shift(1))/(r.rolling(20).std().shift(1)*np.sqrt(10))
rows=[]
for h in [1,3,5,10]:
  ics=[]; n=[]; turns=[]
  f=fac
  fr=np.log(p).shift(-h)-np.log(p)
  for dt in fac.index:
    a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(a)>=8:
      ics.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); n.append(len(a))
  x=np.array(ics); print(h,'dates',len(x),'avgN',np.mean(n),'coverage',np.mean(n)/15,'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0))
# save artifact daily 1d signals
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20330527_risk_adjusted_momentum_signal.csv',index=False)
