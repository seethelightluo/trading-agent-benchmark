import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-12-08')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); ret=r.rolling(10,min_periods=8).sum(); vol=r.rolling(40,min_periods=30).std()*np.sqrt(10)
# Short-horizon contrarian signal, damped for illiquid/noisy names by relative recent range.
rangevol=(P.rolling(20,min_periods=15).max()-P.rolling(20,min_periods=15).min())/P
sig=(-ret/(vol+1e-12))/(1+rangevol.clip(0,1)); sig=sig.shift(1)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20321209_short_reversal_liquidity_signal.csv',index=False)
for H in [10,20,30]:
 y=P.shift(-H)/P-1; vals=[]; ns=[]; dates=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic; vals.append(z);ns.append(len(q));dates.append(d)
   if prev is not None: turns.append(np.nanmean(np.abs(sig.loc[d].rank(pct=True)-prev.rank(pct=True))))
   prev=sig.loc[d]
 x=np.array(vals); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(252),4),'hit',round((x>0).mean(),4),'turn',round(np.mean(turns),4),'coverage',round(sig.notna().mean().mean(),4))
 for label,lo in [('2028','2028-01-01'),('recent365','2031-12-08'),('recent120','2032-06-12')]:
  a=np.array([v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]); print(label,'n',len(a),'IC',round(a.mean(),6) if len(a) else np.nan,'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(252),4) if len(a)>1 else np.nan)
