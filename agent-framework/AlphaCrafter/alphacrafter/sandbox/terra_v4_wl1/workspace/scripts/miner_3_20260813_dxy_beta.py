import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(); m=macro.pct_change()
# DXY defensive sensitivity: negative rolling beta to DXY, estimated only through date t
cov=R.rolling(60,min_periods=45).cov(m); var=m.rolling(60,min_periods=45).var()
f=-cov.div(var,axis=0)
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('dates',d[d.h==1].date.nunique(),'avgN',d[d.h==1].n.mean(),'coverage',d[d.h==1].n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print('h',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
