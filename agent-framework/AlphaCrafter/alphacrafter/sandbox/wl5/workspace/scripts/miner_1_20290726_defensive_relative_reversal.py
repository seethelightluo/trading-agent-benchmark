import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-07-25'); P={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 P[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
# Defensive-relative reversal: reverse each asset's 60d return after removing
# the contemporaneous 60d return of a defensive basket (gold and yields), then
# risk-normalize by causal 20d volatility and percentile-rank cross-sectionally.
defensive=r[['XAU','US10Y','CN10Y']].rolling(3).mean().pct_change(60) if False else r[['XAU','US10Y','CN10Y']].sum(axis=1)
def60=p[['XAU','US10Y','CN10Y']].pct_change(60).mean(axis=1)
resid=p.pct_change(60).sub(def60,axis=0)
sig=(-resid/(r.rolling(20).std()*np.sqrt(252))).rank(axis=1,pct=True)
fr=p.shift(-10)/p-1; vals=[]; ds=[]; ns=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(c): vals.append(c); ds.append(dt); ns.append(len(a))
x=pd.Series(vals,index=ds)
print('DATES',len(x),'AVG_N',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(),'HIT',(x>0).mean(),'COVERAGE',np.mean(ns)/15)
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-07-12','2029-07-25')]:
 z=x.loc[a:b]; print('REG',a,'DATES',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
print('TURNOVER',sig.diff().abs().mean(axis=1).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20290726_defensive_relative_reversal_signal.csv',index=False)
print('ARTIFACT',len(out),'CUTOFF',out.date.max())
