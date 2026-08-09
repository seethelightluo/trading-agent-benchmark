import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; A={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  q=pd.read_csv(f);q.date=pd.to_datetime(q.date);A[s]=q.set_index('date').close.sort_index()
# date-aligned cross-sectional relative 20d momentum
P=pd.concat({s:p.pct_change(20) for s,p in A.items()},axis=1)
F=P.sub(P.median(axis=1),axis=0)
Y=pd.concat({s:p.shift(-1)/p-1 for s,p in A.items()},axis=1)
rows=[]
for d in P.index:
 q=pd.DataFrame({'f':F.loc[d],'y':Y.loc[d]}).dropna()
 if len(q)>=8: rows.append((d,len(q),q.f.corr(q.y)))
x=pd.DataFrame(rows,columns=['date','n','ic']); x=x[x.date<='2026-07-15']; print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 q=x[(x.date.dt.year>=a)&(x.date.dt.year<=b)]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
# turnover by rank top/bottom signal
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
