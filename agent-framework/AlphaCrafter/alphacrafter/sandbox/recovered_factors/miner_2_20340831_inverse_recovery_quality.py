import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-08-30')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for a in A}
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); P=pd.DataFrame({a:D[a].reindex(idx) for a in A}).loc[:E]; R=P.pct_change(fill_method=None)
r20=P/P.shift(20)-1; down=R.where(R<0).rolling(20,min_periods=12).std(); dd=P/P.rolling(60,min_periods=40).max()-1
# Risk-aware contrarian: favor recent losers, scaled by downside volatility, with drawdown sleeve.
F=(-(r20/(down+1e-6)+0.5*dd)).shift(1)
print('cutoff',E.date(),'rows',len(P),'assets',len(A),'coverage',F.notna().mean().mean(),'cells',int(F.notna().sum().sum()))
for h in [1,5,10,20]:
 xs=[];ns=[];dates=[]
 for j in range(len(P)-h):
  z=pd.concat([F.iloc[j].rename('f'),(P.iloc[j+h]/P.iloc[j]-1).rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: xs.append(spearmanr(z.f,z.r).statistic);ns.append(len(z));dates.append(P.index[j])
 x=np.array(xs); print('h',h,'dates',len(x),'meanN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0))
 if h==10:
  for lo,hi in [('2020','2025'),('2025','2030'),('2030','2035')]:
   y=x[[lo<=str(d.year)<hi for d in dates]]; print('regime',lo,hi,'dates',len(y),'IC',np.mean(y) if len(y) else np.nan,'ICIR',np.mean(y)/np.std(y,ddof=1) if len(y)>1 else np.nan)
r=F.rank(axis=1,pct=True); q=[]
for j in range(1,len(r)):
 z=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(z)>=8:q.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',np.nanmean(q),'turnover_dates',len(q)); print('library_audit NOT COMPUTED')
