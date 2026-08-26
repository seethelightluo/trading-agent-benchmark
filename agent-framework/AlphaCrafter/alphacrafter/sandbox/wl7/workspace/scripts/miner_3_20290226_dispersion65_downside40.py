import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-02-25'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy();d.date=pd.to_datetime(d.date).dt.normalize();P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); ret40=P/P.shift(40)-1
neg2=(r.clip(upper=0)**2).rolling(40,min_periods=25).mean(); down=np.sqrt(neg2)*np.sqrt(252)
base=(ret40/(down+0.01)).replace([np.inf,-np.inf],np.nan).shift(1)
r5=P/P.shift(5)-1; disp=r5.std(axis=1); threshold=disp.rolling(120,min_periods=60).quantile(.65).shift(1); gate=(disp>threshold)
f=base.where(gate, np.nan)
ics=[]; ns=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): ics.append((P.index[i],c));ns.append(ok.sum())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True);b=f.iloc[i].rank(pct=True);ok=a.notna()&b.notna()
 if ok.sum()>=8: turn.append(np.abs(a[ok]-b[ok]).mean())
print('data_start',P.index.min(),'data_end',P.index.max(),'valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'active_fraction',gate.mean(),'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turn))
for label,lo,hi in [('2020_24','2020-01-01','2025-01-01'),('2025_26','2025-01-01','2027-01-01'),('2027_29','2027-01-01','2029-02-26'),('recent','2028-09-01','2029-02-26')]:
 w=z[(z.index>=lo)&(z.index<hi)];print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan,'hit',np.mean(w>0) if len(w) else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_3_20290226_dispersion65_downside40_signal.csv',index=False)
