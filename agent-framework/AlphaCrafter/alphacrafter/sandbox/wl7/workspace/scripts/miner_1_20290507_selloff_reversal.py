import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-05-06'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); ret10=P.pct_change(10); breadth=(r>0).mean(axis=1)
# Selloff-only, extreme breadth-conditioned relative reversal; lag all inputs.
for th in [.15,.25,.35]:
 state=(-2*(.5-breadth)).where(breadth<=.5-th,0).shift(1)
 f=ret10.sub(ret10.mean(axis=1),axis=0).mul(-state,axis=0).shift(1)
 ics=[]; ns=[]
 for i in range(len(P)-10):
  x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   c=x[ok].corr(y[ok],method='spearman')
   if np.isfinite(c): ics.append((P.index[i],c)); ns.append(ok.sum())
 z=pd.Series(dict(ics)); print('threshold',th,'dates',len(z),'avg_n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'coverage',np.mean(ns)/15)
 if th==.25: f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_1_20290507_selloff_reversal_signal.csv',index=False)
