import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-04-08'); P={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); mr=r.mean(axis=1)
# Market-neutral medium-term reversal: reverse asset cumulative return in excess of equal-weight market over 20d.
f=-(P.pct_change(20).sub(mr.rolling(20).sum(),axis=0)); f=f.sub(f.mean(axis=1),axis=0)
ics=[];ns=[]
for i in range(len(P)-20):
 x=f.iloc[i]; y=P.iloc[i+20]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): ics.append((P.index[i],c)); ns.append(ok.sum())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turn.append(np.abs(a[ok]-b[ok]).mean())
print('data_start',P.index.min(),'data_end',P.index.max(),'rows',len(P),'valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turn))
for label,lo,hi in [('2020_24','2020-01-01','2025-01-01'),('2025_26','2025-01-01','2027-01-01'),('2027_28','2027-01-01','2029-01-01'),('recent','2028-09-01','2030-01-01')]:
 w=z[(z.index>=lo)&(z.index<hi)]; print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan,'hit',np.mean(w>0) if len(w) else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20290409_residual_reversal20_signal.csv',index=False)
