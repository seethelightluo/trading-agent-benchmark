import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
O={}; C={}
for s in A:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 O[s]=d['open']; C[s]=d['close']
op=pd.concat(O,axis=1).sort_index(); cl=pd.concat(C,axis=1).sort_index().loc[:'2033-04-13']; op=op.reindex(cl.index)
# fade 5-day average overnight gap, lagged naturally as signal at completed close
prev=cl.shift(1); gap=op/prev-1
sig=-gap.rolling(5,min_periods=3).mean()
print('overnight gap reversal; instruments',len(A),'last',cl.index[-1].date())
for h in [5,10,20,40]:
 f=cl.shift(-h).div(cl)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(cl)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(cl.index[i])
 x=np.asarray(vals); ser=pd.Series(x,index=pd.DatetimeIndex(dates))
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4),'annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
