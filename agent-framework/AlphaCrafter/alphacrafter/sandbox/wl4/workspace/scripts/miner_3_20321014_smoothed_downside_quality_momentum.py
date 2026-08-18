import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-10-13')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); ret40=P.pct_change(40)
# slower downside-risk normalization and slower consistency, all lagged
D=(r.clip(upper=0)**2).rolling(90,min_periods=60).mean().pow(.5)
C=(r.gt(0).rolling(60,min_periods=45).mean()-0.5)*2
sig=(ret40/(D*np.sqrt(40))).mul(C).shift(1)
y=P.shift(-30)/P-1
vals=[]; ns=[]; turns=[]; prev=None; dates=[]
for d in sig.index:
 q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
  vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(d)
  if prev is not None:
   a=sig.loc[d].reindex(U); b=prev.reindex(U)
   turns.append(np.nanmean(np.abs((a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12))))
  prev=sig.loc[d]
x=np.asarray(vals); print('dates',len(x),'avgN',np.mean(ns),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12)*np.sqrt(252),'hit',np.mean(x>0),'turn',np.nanmean(turns),'coverage',sig.notna().mean().mean())
for label,lo in [('early','2020-01-01'),('mid','2028-01-01'),('recent','2031-09-01')]:
 z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]
 print(label,len(z),np.nanmean(z) if z else np.nan)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_3_20321014_smoothed_downside_quality_signal.csv',index=False)
