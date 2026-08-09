import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close for s,d in D.items()}); V=pd.DataFrame({s:d.volume for s,d in D.items()})
R=P.pct_change(); # volume-confirmed trend: return-volume covariance / vol(volume), signed association
# use rolling correlation of returns and log volume changes, completed data only
LV=np.log(V.replace(0,np.nan)).diff()
f=R.rolling(20,min_periods=15).corr(LV)
# forward close returns
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt], P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('factor=return/log-volume-change rolling corr; dates',d[d.h==1].date.nunique(),'avgN',d[d.h==1].n.mean(),'coverage',d[d.h==1].n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print('h',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252)
)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('valid dates',len(d[d.h==1]))
