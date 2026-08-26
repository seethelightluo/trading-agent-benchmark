import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-04-22'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); ret10=P.pct_change(10); breadth=(r>0).mean(axis=1)
state=((breadth-0.5)*2).shift(1)
# align the scalar state by date explicitly, then lag the completed signal once
f=ret10.sub(ret10.mean(axis=1),axis=0).mul(-state,axis=0).shift(1)
ics=[]; ns=[]; turns=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): ics.append((P.index[i],c)); ns.append(ok.sum())
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turns.append(np.abs(a[ok]-b[ok]).mean())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for label,lo,hi in [('2020_22','2020-01-01','2023-01-01'),('2023_25','2023-01-01','2026-01-01'),('2026_28','2026-01-01','2029-01-01'),('recent','2028-09-01','2029-04-23')]:
 w=z[(z.index>=lo)&(z.index<hi)]; print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_1_20290423_breadth_reversal_signal.csv',index=False)
print('dates_used',P.index.min(),P.index.max(),'instruments',len(P.columns))
