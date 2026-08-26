import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2028-10-22'); P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); f=-(P/P.shift(10)-1)/(r.rolling(30,min_periods=20).std()*np.sqrt(20)+1e-8)
def calc(h):
 vals=[]; ns=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   z=x[ok].corr(y[ok],method='spearman')
   if np.isfinite(z): vals.append((P.index[i],z)); ns.append(ok.sum())
 q=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); a=q.ic
 print('horizon',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for lo,hi in [(2025,2026),(2027,2028)]:
  z=a[(a.index.year>=lo)&(a.index.year<=hi)]; print('regime',lo,hi,len(z),z.mean(),z.mean()/z.std(ddof=1))
 return q
q=calc(20); out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20281023_volnorm_reversal10_signal.csv',index=False)
