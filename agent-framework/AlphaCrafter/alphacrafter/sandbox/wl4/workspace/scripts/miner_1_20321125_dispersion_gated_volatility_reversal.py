import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-11-24')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); vol=r.rolling(60,min_periods=40).std()
# Contrarian residual reversal, activated only when cross-sectional dispersion is above its trailing median.
ret20=P.pct_change(20); disp=ret20.std(axis=1); gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
raw=-(ret20/(vol*np.sqrt(20)))
sig=raw.rank(axis=1,pct=True)-0.5
sig=sig.mul(gate,axis=0).shift(1)
for h in (10,20,30):
 y=P.shift(-h)/P-1; ics=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
   if prev is not None: turns.append(np.nanmean(np.abs(sig.loc[d].values-prev.values)))
   prev=sig.loc[d]
 x=np.asarray(ics); sd=np.nanstd(x,ddof=1)
 print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(sd+1e-12),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),6))
 for label,lo in [('2028plus','2028-01-01'),('recent365','2031-05-01'),('recent120','2032-05-01')]:
  z=[]
  for d in sig.index[sig.index>=pd.Timestamp(lo)]:
   q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  print(' ',label,len(z),round(np.nanmean(z),6) if z else np.nan,round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6) if len(z)>1 else np.nan)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20321125_dispersion_gated_volatility_reversal_signal.csv',index=False)
print('artifact_rows',len(out),'coverage',round(sig.notna().mean().mean(),6),'gate_rate',round(gate.mean(),6))
