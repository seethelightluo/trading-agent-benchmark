import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-11-19'); defensive={'XAU','US10Y','CN10Y'}
P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); vol=r.rolling(30,min_periods=20).std()
raw=P.pct_change(20)/(vol*np.sqrt(20)); breadth=(P.pct_change(20)>0).mean(axis=1); bear=breadth<0.5
dflag=pd.Series([1.0 if s in defensive else 0.0 for s in P.columns],index=P.columns)
# In bearish breadth regimes, prefer assets with weaker recent risk-adjusted returns
# (short-horizon cross-asset reversal), while retaining a defensive tilt.
f=(-raw).sub((-raw).mean(axis=1),axis=0).add(bear.astype(float).mul(0.5).values[:,None]*dflag.values,axis=1)
ics=[]; ns=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(z): ics.append((P.index[i],z)); ns.append(ok.sum())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); ic=q.ic
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turn.append(np.abs(a[ok]-b[ok]).mean())
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'bear_frac',bear.mean())
print('IC',ic.mean(),'ICIR_daily',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0),'turnover',np.mean(turn))
for a,b in [(2020,2022),(2023,2025),(2026,2028)]:
 z=ic[(ic.index.year>=a)&(ic.index.year<=b)]; print('regime',a,b,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20281120_defensive_relative_reversal_signal.csv',index=False)
