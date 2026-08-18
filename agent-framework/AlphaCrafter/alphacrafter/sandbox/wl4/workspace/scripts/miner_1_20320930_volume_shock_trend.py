import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-09-29')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
V=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['volume'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); r20=P.pct_change(20); vol60=r.rolling(60,min_periods=40).std()
# Trend signal strengthened by an observable, lagged volume participation shock; bounded to limit crypto/commodity outliers.
shock=(V/(V.rolling(60,min_periods=40).median()+1e-12)).clip(0.25,4.0)
sig=(r20/(vol60*np.sqrt(20))*np.log(shock)).shift(1)
y={h:P.shift(-h)/P-1 for h in [10,20,30]}
for h in [10,20,30]:
 vals=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
   if prev is not None:
    a=sig.loc[d].reindex(U); b=prev.reindex(U); turns.append(np.nanmean(np.abs(a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12)))
   prev=sig.loc[d]
 x=np.asarray(vals); print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.nanmean(x),'dailyICIR',np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),'annualICIR',np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12)*np.sqrt(252),'hit',np.mean(x>0),'turn',np.nanmean(turns))
 for label,lo in [('2026-2029','2026-01-01'),('2030-2032','2030-01-01'),('recent365','2031-04-01')]:
  z=[]
  for d in sig.index[sig.index>=pd.Timestamp(lo)]:
   q=pd.concat([sig.loc[d],y[h].loc[d]],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  print(' ',label,len(z),np.nanmean(z) if z else np.nan)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20320930_volume_shock_trend_signal.csv',index=False)
print('artifact',len(out),'coverage',sig.notna().mean().mean())
