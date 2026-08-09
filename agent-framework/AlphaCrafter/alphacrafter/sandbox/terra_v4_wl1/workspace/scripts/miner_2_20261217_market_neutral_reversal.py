import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d['close'] for s,d in D.items()}); R=P.pct_change()
# Market-neutral short-term reversal: negate each asset's 5d return after subtracting same-day cross-sectional mean.
f=-(P.pct_change(5).sub(P.pct_change(5).mean(axis=1),axis=0))
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  x=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: rows.append((dt,h,len(x),spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('factor market-neutral 5d reversal; dates',df[df.h==1].date.nunique(),'avgN',df[df.h==1].n.mean(),'coverage',df[df.h==1].n.mean()/15)
for h in [1,5,10]:
 q=df[df.h==h].ic; print('h',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
