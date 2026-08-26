import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Smooth multi-horizon recovery: average normalized drawdowns from 30/60 session highs,
# conditioned on positive 120-session trend. Lagged before forward-return measurement.
vol=r.rolling(20,min_periods=15).std()
dd30=1-p/p.rolling(30,min_periods=24).max(); dd60=1-p/p.rolling(60,min_periods=48).max()
trend=p.pct_change(120)
raw=-(0.6*dd30+0.4*dd60)/(vol*np.sqrt(20)); raw=raw.where(trend>0)
sig=raw.shift(1); rows=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); cov.append(sig.loc[dt].notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); v=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(cov),'IC',v.mean(),'dailyICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
ranks=sig.rank(axis=1,pct=True); tt=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: tt.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(tt))
for w in [365,750,1260]:
 q=v.tail(w); print('recent',w,'ICIR',q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr))
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20341123_smoothed_drawdown_reversal_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_3_20341123_smoothed_drawdown_reversal_ic.csv',index=False)
