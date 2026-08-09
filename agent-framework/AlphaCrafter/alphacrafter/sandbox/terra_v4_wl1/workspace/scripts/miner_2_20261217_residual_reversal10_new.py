import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close for s,d in D.items()}); R=P.pct_change()
# Asset-relative medium-horizon reversal: negate 10d return after removing cross-sectional market component.
r10=P.pct_change(10); f=-(r10.sub(r10.mean(axis=1),axis=0))
rows=[]
for dt in f.index:
 x=pd.concat([f.loc[dt],P.shift(-1).loc[dt]/P.loc[dt]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
  rows.append((dt,len(x),spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']); q=df.ic.dropna()
print('dates',len(q),'avgN',df.n.mean(),'coverage',df.n.mean()/15,'IC',q.mean(),'ICIR_daily',q.mean()/q.std(ddof=1),'ICIR_ann',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=df[(df.date>=a)&(df.date<=b)].ic.dropna();print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
# decay
for h in [5,10]:
 z=[]
 for dt in f.index:
  x=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('h',h,'dates',len(z),'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
