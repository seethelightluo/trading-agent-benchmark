import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2029-03-11'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize()
  P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change()
# Drawdown-recovery asymmetry: lagged 60d drawdown, rewarded by recent 10d recovery,
# with recovery scaled by 20d volatility. Cross-sectional ranks are used for robustness.
peak=P.rolling(60,min_periods=40).max(); dd=P/peak-1
rec=P.pct_change(10); vol=r.rolling(20,min_periods=15).std()
f=((rec/(vol*np.sqrt(10))) * (-dd.clip(upper=0))).shift(1)
ics=[]; ns=[]; turns=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): ics.append((P.index[i],c));ns.append(ok.sum())
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turns.append(np.abs(a[ok]-b[ok]).mean())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for label,lo,hi in [('2025_26','2025-01-01','2027-01-01'),('2027_29','2027-01-01','2029-03-12'),('recent','2028-09-01','2029-03-12')]:
 w=z[(z.index>=lo)&(z.index<hi)]; print(label,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_3_20290312_drawdown_recovery_signal.csv',index=False)
print('dates_used',P.index.min(),P.index.max(),'instruments',len(P.columns))
