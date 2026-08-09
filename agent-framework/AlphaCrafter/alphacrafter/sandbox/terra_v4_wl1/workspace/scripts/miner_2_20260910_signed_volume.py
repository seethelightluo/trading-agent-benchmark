import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
Ds={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
V=pd.DataFrame({s:Ds[s]['volume'] for s in U}).sort_index(); P=pd.DataFrame({s:Ds[s]['close'] for s in U}).sort_index()
# volume shock confirmed by return direction: signed volume surprise, trailing only
rv=V/V.rolling(20,min_periods=15).median()-1
f=np.sign(P.pct_change())*rv
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic']); q1=d[d.h==1].ic
print('candidate=signed_volume_shock_20d','dates',d[d.h==1].date.nunique(),'avgN',q1.size and d[d.h==1].n.mean(),'coverage',d[d.h==1].n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print('h',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31')]:
 q=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
