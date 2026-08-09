import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change()
# downside semideviation: RMS of negative returns over trailing 30 completed observations
neg2=R.clip(upper=0).pow(2); cnt=(R<0).rolling(30).sum(); f=-np.sqrt(neg2.rolling(30).sum()/cnt.where(cnt>=8))
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic']); q=d[d.h==1]; print('dates',q.date.nunique(),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print('h',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31')]:
 q=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print('regime',a[:4],len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
