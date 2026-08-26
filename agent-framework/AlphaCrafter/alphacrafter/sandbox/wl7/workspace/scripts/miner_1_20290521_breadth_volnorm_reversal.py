import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-05-20'); P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); ret10=P.pct_change(10); vol20=r.rolling(20,min_periods=15).std(); breadth=(r>0).mean(axis=1); bs=breadth.rolling(5,min_periods=5).mean(); state=(-2*(.5-bs)).where(bs<=.25,0).shift(1)
raw=ret10.div(vol20*np.sqrt(10)); f=raw.sub(raw.mean(axis=1),axis=0).mul(-state,axis=0).shift(1)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_1_20290521_breadth_volnorm_reversal_signal.csv',index=False)
ics=[]; ns=[]; by=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=P.iloc[i+10]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  c=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(c): ics.append(c); ns.append(ok.sum()); by.append((P.index[i],c))
z=pd.Series(ics); print('rows',len(P),'start',P.index.min(),'end',P.index.max(),'dates',len(z),'avg_n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'coverage',np.mean(ns)/15)
print('active',int((state!=0).sum()),'breadth_min',bs.min(),'q',bs.quantile([.01,.05,.1,.25]).to_dict())
for h in [5,15,20]:
 a=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   q=x[ok].corr(y[ok],method='spearman')
   if np.isfinite(q): a.append(q)
 q=pd.Series(a); print('h',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
