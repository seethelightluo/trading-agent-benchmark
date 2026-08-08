import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
p=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:'2033-10-12']; r=p.pct_change()
# Volatility-neutralized medium reversal: reversal divided by recent vol, with a slow-vol penalty.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
sig=-(p/p.shift(20)-1)/(v20*(1+v20/(v60+1e-12))+1e-12); sig=sig.shift(1)
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A),'dates',len(p),'cells',int(sig.notna().sum().sum()),'coverage',round(sig.notna().sum().sum()/sig.size,6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>=3:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x):z.append(x);ds.append(dt);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),3),'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12),np.mean(z>0)))
 for lab,lo,hi in [('2020-23','2020','2023'),('2024-27','2024','2027'),('2028-30','2028','2030'),('2031-33','2031','2033')]:
  q=z[(ds>=pd.Timestamp(lo+'-01-01'))&(ds<=pd.Timestamp(hi+'-12-31'))]
  print(' regime',lab,'n',len(q),'IC/ICIR',('%.8f/%.8f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)) if len(q)>1 else 'NA'))
rank=sig.rank(axis=1,pct=True);print('turn10',round(float(rank.diff(10).abs().mean(axis=1).dropna().mean()),6))
# latest rolling blocks
for end in ['2027-12-31','2030-12-31','2033-10-12']:
 z=[]; ds=[]; f=p.shift(-10)/p-1
 for dt in sig.index[sig.index<=end]:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(dt)
 z=np.array(z[-504:]);print('rollingH10',end,'n',len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12)))
print('LIBRARY_AUDIT_REQUIRED: candidate not admitted because exact pooled max absolute Spearman correlation against every active library signal was not reconstructed.')
