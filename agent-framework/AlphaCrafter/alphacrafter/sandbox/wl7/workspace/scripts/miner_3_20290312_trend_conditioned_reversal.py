import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-03-11'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy();d.date=pd.to_datetime(d.date).dt.normalize();P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index();r=P.pct_change();v=r.rolling(20,min_periods=15).std()
# Residual short-term reversal inside a medium-term trend: 5d reversal, scaled by vol,
# multiplied by the sign of lagged 60d momentum. All inputs lagged one session.
trend=np.sign(P.pct_change(60)); f=(-P.pct_change(5)/(v*np.sqrt(5))*trend).shift(1)
ics=[];ns=[];turn=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c):ics.append((P.index[i],c));ns.append(ok.sum())
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True);b=f.iloc[i].rank(pct=True);ok=a.notna()&b.notna()
 if ok.sum()>=8:turn.append(abs(a[ok]-b[ok]).mean())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date');z=q.ic
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turn))
for label,lo,hi in [('2025_26','2025-01-01','2027-01-01'),('2027_29','2027-01-01','2029-03-12'),('recent','2028-09-01','2029-03-12')]:
 w=z[(z.index>=lo)&(z.index<hi)];print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_3_20290312_trend_conditioned_reversal_signal.csv',index=False)
print('dates_used',P.index.min(),P.index.max(),'instruments',len(P.columns))
