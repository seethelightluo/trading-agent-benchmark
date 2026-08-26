import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-01-28'); P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Multi-horizon risk-adjusted momentum: blend 20d and 60d returns, normalized by recent volatility.
f=.6*P.pct_change(20)/(vol*np.sqrt(20))+.4*P.pct_change(60)/(vol*np.sqrt(60)); f=f.sub(f.mean(axis=1),axis=0)
for H in [10,20]:
  ics=[]; ns=[]
  for i in range(len(P)-H):
   x=f.iloc[i]; y=P.iloc[i+H]/P.iloc[i]-1; ok=x.notna()&y.notna()
   if ok.sum()>=8 and x[ok].nunique()>1:
    z=x[ok].corr(y[ok],method='spearman')
    if np.isfinite(z):ics.append((P.index[i],z));ns.append(ok.sum())
  q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
  print('H',H,'valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
  for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028),(2028,2029)]:
   w=z[(z.index.year>=a)&(z.index.year<=b)];print('regime',a,b,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
# rank turnover
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True);b=f.iloc[i].rank(pct=True);ok=a.notna()&b.notna()
 if ok.sum()>=8:turn.append(abs(a[ok]-b[ok]).mean())
print('turnover',np.mean(turn))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20290129_multihorizon_mom_signal.csv',index=False)
