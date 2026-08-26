import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-03-25'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); peak=P.rolling(60,min_periods=20).max(); dd=(P/peak-1).clip(upper=0); rec=P.pct_change(10); vol=r.rolling(20,min_periods=10).std(); f=((-dd*rec)/(vol*np.sqrt(10))).shift(1)
rows=[]; ns=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): rows.append((P.index[i],c)); ns.append(ok.sum())
q=pd.DataFrame(rows,columns=['date','ic']).set_index('date'); z=q.ic; turns=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turns.append(np.abs(a[ok]-b[ok]).mean())
print('valid_dates',len(z),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for label,lo,hi in [('2025_26','2025-01-01','2027-01-01'),('2027_28','2027-01-01','2029-01-01'),('2029','2029-01-01','2029-03-26'),('recent','2028-09-01','2029-03-26')]:
 w=z[(z.index>=lo)&(z.index<hi)]; print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20290326_drawdown_recovery_signal.csv',index=False)
print('dates_used',P.index.min(),P.index.max(),'instruments',len(P.columns))
