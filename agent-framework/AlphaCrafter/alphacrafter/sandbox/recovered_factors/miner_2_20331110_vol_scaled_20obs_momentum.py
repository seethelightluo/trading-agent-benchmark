import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
p=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:'2033-11-09']; r=p.pct_change()
# One interpretable candidate: volatility-scaled medium-term momentum, lagged one day.
vol=r.rolling(20,min_periods=15).std(); sig=((p/p.shift(20)-1)/(vol*np.sqrt(20)+1e-12)).shift(1)
print('candidate=vol_scaled_20obs_momentum','period',p.index.min().date(),p.index.max().date(),'dates',len(p),'assets',len(A),'coverage',round(sig.notna().sum().sum()/sig.size,6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>=3:
   x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(x): z.append(x); ds.append(dt); ns.append(len(q))
 z=np.array(z); ds=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),8),'ICIR',round(z.mean()/(z.std(ddof=1)+1e-12),8),'hit',round(np.mean(z>0),4))
 for lab,lo,hi in [('2024-27','2024','2027'),('2028-30','2028','2030'),('2031-33','2031','2033')]:
  q=z[(ds>=pd.Timestamp(lo+'-01-01'))&(ds<=pd.Timestamp(hi+'-12-31'))]
  print(' regime',lab,'n',len(q),'IC',round(q.mean(),8) if len(q) else None,'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12),8) if len(q)>1 else None)
print('turn10',round(float(sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).dropna().mean()),6))
print('admission_status=BLOCKED; exact pooled Spearman correlation against every admitted factor was not reconstructed')
