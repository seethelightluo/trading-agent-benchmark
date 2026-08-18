import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2033-08-31')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:CUT] for s in U}).sort_index()
R=P.pct_change(); bm=R.median(axis=1); resid=R.sub(bm,axis=0)
# Candidate: 10d residual reversal weighted by each asset's relative idiosyncratic volatility.
rv=resid.rolling(30,min_periods=20).std(); relvol=rv.div(rv.median(axis=1),axis=0).clip(0.5,2.0)
sig=(-(resid.rolling(10,min_periods=8).sum())*relvol).shift(1)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(U))
for H in [1,5,10,20]:
 y=P.shift(-H).div(P)-1;ics=[];ns=[];turn=[];prev=None; ds=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic;ics.append(z);ns.append(len(q));ds.append(d)
   rk=q.iloc[:,0].rank(pct=True)
   if prev is not None: turn.append(np.mean(abs(rk-prev)))
   prev=rk
 a=np.asarray(ics); print('H',H,'obs',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(np.mean(turn),5))
 if H==1:
  for lo,hi in [(2020,2024),(2025,2028),(2029,2031),(2032,2033)]:
   z=a[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)]; print(' regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
 for n in [260,520,780]:
  z=a[-n:];print(' recent',n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('signal_artifact')
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/artifacts/miner_2_20330901_idiov_reversal_signal.csv',index=False)
