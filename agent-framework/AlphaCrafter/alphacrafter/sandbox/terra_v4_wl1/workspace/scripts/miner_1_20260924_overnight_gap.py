import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2026-09-24')
D={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); O=pd.DataFrame({s:D[s].open for s in U}).sort_index()
# one-day overnight gap reversal: completed open relative to prior close, sign inverted
G=-(O/P.shift(1)-1)
rows=[]
for dt in G.index:
 for h in [1,3,5,10]:
  z=pd.concat([G.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('dates',df[df.h==1].date.nunique(),'avgN',df[df.h==1].n.mean(),'coverage',df[df.h==1].n.mean()/15)
for h in [1,3,5,10]:
 q=df[df.h==h].ic; print('H',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-09-24')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic; print('regime',a,b,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('turnover',G.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('recent',[(end, (q:=df[(df.h==1)&(df.date>=pd.Timestamp(end)-pd.Timedelta(days=365))&(df.date<=end)].ic).mean(),q.mean()/q.std(ddof=1)) for end in ['2024-12-31','2025-12-31','2026-09-24']])
