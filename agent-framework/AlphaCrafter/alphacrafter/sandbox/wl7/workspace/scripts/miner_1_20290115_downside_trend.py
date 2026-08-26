import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-01-14')
P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change()
# downside-risk-adjusted trend: 40d return, penalize downside deviation, gated by positive 20d breadth
ret40=P.shift(1)/P.shift(41)-1
down=r.where(r<0,0).rolling(40,min_periods=25).std()
f=-ret40/(down*np.sqrt(40)+1e-8)
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1)
f=f.where(breadth>0.5)
y=P.shift(-10)/P-1
ics=[]; ns=[]; dates=[]
for i,d in enumerate(P.index):
 x=f.iloc[i]; yy=y.iloc[i]; ok=x.notna()&yy.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(yy[ok],method='spearman')
  if np.isfinite(z):ics.append(z);ns.append(ok.sum());dates.append(d)
q=pd.Series(ics,index=pd.DatetimeIndex(dates)); print('dates',len(q),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC10',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028),(2029,2029)]:
 w=q[(q.index.year>=a)&(q.index.year<=b)]; print('regime',a,b,'n',len(w),'ic',w.mean(),'icir',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
turn=[]; prev=None
for d in P.index:
 x=f.loc[d]
 if x.notna().sum()>=8:
  if prev is not None:
   ok=x.notna()&prev.notna()
   if ok.sum()>=8: turn.append(np.abs(x[ok].rank(pct=True)-prev[ok].rank(pct=True)).mean())
  prev=x
print('turnover',np.mean(turn),'turn_dates',len(turn),'assets',len(P.columns))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290115_downside_trend_signal.csv',index=False)
