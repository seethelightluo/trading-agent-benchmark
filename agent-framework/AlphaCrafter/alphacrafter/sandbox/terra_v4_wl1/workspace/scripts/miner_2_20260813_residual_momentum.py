import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2026-07-15']; R=P.pct_change(); b=R['SPX'].shift(1); rr=R.shift(1)
beta=pd.DataFrame(index=P.index)
for s in U: beta[s]=rr[s].rolling(60,min_periods=45).cov(b)/b.rolling(60,min_periods=45).var()
f=(P/P.shift(20)-1)-beta.mul((P['SPX']/P['SPX'].shift(20)-1),axis=0)
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic']); d['date']=pd.to_datetime(d['date'])
print('dates',d[d.h==1].date.nunique(),'avgN',d[d.h==1].n.mean(),'coverage',d[d.h==1].n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print('H',h,'N',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,bnd in [(2020,2022),(2023,2024),(2025,2026)]:
 q=d[(d.h==1)&(d.date.dt.year.between(a,bnd))].ic; print('REG',a,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
q=d[(d.h==1)&(d.date>='2025-01-01')].ic; print('recent',q.mean(),q.mean()/q.std()*np.sqrt(252))
