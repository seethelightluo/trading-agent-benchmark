import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-09-24')
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index(); P=P[P.index<=cut]; R=P.pct_change()
# Novel interpretable signal: acceleration of 5d momentum versus 20d momentum, smoothed cross-sectionally
f=(P/P.shift(5)-1)-(P/P.shift(20)-1)/4
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic'])
q=d[d.h==1]
print('candidate acceleration_5v20','range',P.index.min(),P.index.max(),'dates',q.date.nunique(),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
for h in [1,5,10]:
 x=d[d.h==h].ic; print('h',h,'obs',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-09-24')]:
 x=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print('regime',a,b,'obs',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('recent252',q[q.date>=q.date.max()-pd.Timedelta(days=365)].ic.mean(),q[q.date>=q.date.max()-pd.Timedelta(days=365)].ic.std(ddof=1))
# Save compact signal artifact for audit
f.to_csv('scripts/miner_2_20260924_acceleration_signal.csv')
