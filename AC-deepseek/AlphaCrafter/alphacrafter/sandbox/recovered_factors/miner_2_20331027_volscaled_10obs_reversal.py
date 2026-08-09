import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 return d.close.astype(float)
p=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:'2033-10-26']; r=p.pct_change()
# Candidate: volatility-scaled 10-observation reversal, lagged one completed day.
v=r.rolling(20,min_periods=15).std(); sig=-(p/p.shift(10)-1)/(v*np.sqrt(10)+1e-12); sig=sig.shift(1)
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A),'dates',len(p),'cells',int(sig.notna().sum().sum()),'coverage',round(sig.notna().sum().sum()/sig.size,6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>=3:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x): z.append(x); ds.append(dt); ns.append(len(q))
 z=np.array(z); ds=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),3),'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12),np.mean(z>0)))
 for lab,lo,hi in [('2024-27','2024','2027'),('2028-30','2028','2030'),('2031-33','2031','2033')]:
  q=z[(ds>=pd.Timestamp(lo+'-01-01'))&(ds<=pd.Timestamp(hi+'-12-31'))]
  print(' regime',lab,'n',len(q),'IC/ICIR',('%.8f/%.8f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)) if len(q)>1 else 'NA'))
rank=sig.rank(axis=1,pct=True); print('turn10',round(float(rank.diff(10).abs().mean(axis=1).dropna().mean()),6))
print('LIBRARY_AUDIT_REQUIRED: exact pooled max library correlation not reconstructed; no admission')
