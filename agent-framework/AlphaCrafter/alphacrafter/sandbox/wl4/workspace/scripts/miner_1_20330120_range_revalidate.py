import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2033-01-19')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
lo=P.rolling(60,min_periods=40).min(); hi=P.rolling(60,min_periods=40).max(); sig=(.5-(P-lo)/(hi-lo+1e-12)).shift(1)
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/artifacts/miner_1_20330120_range_location_reversal_60d_signal.csv',index=False)
y=P.shift(-10)/P-1; vals=[]; ns=[]; dates=[]; turns=[]; prev=None
for d in sig.index:
 q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
  vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));dates.append(d)
  if prev is not None:
   a=sig.loc[d].reindex(U);b=prev.reindex(U);turns.append(np.nanmean(np.abs((a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12))))
  prev=sig.loc[d]
x=np.array(vals); print('dates',len(x),'avgN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1)*np.sqrt(252),'hit',np.mean(x>0),'turn',np.mean(turns),'coverage',sig.notna().mean().mean())
for label,lo_ in [('recent365','2031-08-01'),('recent120','2032-07-01')]:
 z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo_)]; print(label,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1)*np.sqrt(252))
