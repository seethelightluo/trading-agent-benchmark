import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,5000)
  except Exception:d=None
  if d is not None and len(d):break
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.drop_duplicates('date').set_index('date').close
P=pd.DataFrame(D).sort_index();r=P.pct_change(); ret=P.pct_change(20); vol=r.rolling(20,min_periods=10).std()
# medium-term trend, normalized by volatility; lagged one observation
f=(ret/(vol*np.sqrt(20))).shift(1)
rows=[];ns=[]
for i in range(len(P)-10):
 z=pd.concat([f.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):rows.append((P.index[i],c));ns.append(len(z))
q=pd.DataFrame(rows,columns=['date','ic']).set_index('date');a=q.ic; turns=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8:turns.append(np.abs(z.iloc[:,0].rank(pct=True)-z.iloc[:,1].rank(pct=True)).mean())
print('data',P.index.min(),P.index.max(),'dates',len(a),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(turns))
for lab,lo,hi in [('2025_26','2025-01-01','2027-01-01'),('2027_28','2027-01-01','2029-01-01'),('recent','2028-09-01','2029-04-23')]:
 x=a[(a.index>=lo)&(a.index<hi)];print(lab,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20290423_trend_vol_signal.csv',index=False)
