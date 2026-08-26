import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-12-31')
P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); bench=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0)
res5=r.rolling(5,min_periods=5).sum()-beta.mul(bench.rolling(5,min_periods=5).sum(),axis=0)
vol=r.rolling(30,min_periods=20).std(); raw=-res5/(vol*np.sqrt(5)); raw=raw.sub(raw.mean(axis=1),axis=0)
disp=r.rolling(5,min_periods=5).std().mean(axis=1); gate=disp.gt(disp.rolling(120,min_periods=60).quantile(.75).shift(1))
f=raw.where(gate,0.0)
y=P.shift(-20)/P-1
ics=[]; ns=[]; active=[]; dates=[]
for i,d in enumerate(P.index):
 x=f.iloc[i]; yy=y.iloc[i]; ok=x.notna()&yy.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(yy[ok],method='spearman')
  if np.isfinite(z):ics.append(z);ns.append(ok.sum());active.append(gate.iloc[i]);dates.append(d)
q=pd.Series(ics,index=pd.DatetimeIndex(dates)); print('dates',len(q),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'active_fraction',np.mean(active),'IC20',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 w=q[(q.index.year>=a)&(q.index.year<=b)]; print('regime',a,b,'n',len(w),'ic',w.mean(),'icir',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
# rank turnover on active signals only
turn=[]; prev=None
for d in P.index:
 x=f.loc[d];
 if gate.loc[d]:
  if prev is not None:
   ok=x.notna()&prev.notna()
   if ok.sum()>=8: turn.append(np.abs(x[ok].rank(pct=True)-prev[ok].rank(pct=True)).mean())
  prev=x
print('turnover',np.mean(turn) if turn else np.nan,'turn_dates',len(turn))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290101_dispersion_residual5_signal.csv',index=False)
