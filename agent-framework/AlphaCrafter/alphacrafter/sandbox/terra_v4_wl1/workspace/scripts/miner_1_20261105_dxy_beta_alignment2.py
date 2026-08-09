import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';macro='../persistent/index_data';cut=pd.Timestamp('2026-11-05')
P=pd.DataFrame({s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index();P=P.loc[:cut];R=P.pct_change()
D=pd.read_csv(os.path.join(macro,'DXY.csv'),parse_dates=['date']).set_index('date')['close'].astype(float).loc[:cut];mr=D.pct_change()
n=60; minp=40
# per-asset rolling cov/variance on actual overlapping observations
f=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U:
 z=pd.concat([R[s],mr],axis=1,keys=['r','m']).dropna()
 beta=z.r.rolling(n,min_periods=minp).cov(z.m)/z.m.rolling(n,min_periods=minp).var()
 f.loc[beta.index,s]=-beta
print('rows',len(P),'DXY',len(D),'factor valid',f.notna().sum().sum(),'coverage',f.notna().sum().sum()/f.size)
for h in [1,3,5,10]:
 out=[]
 for dt in R.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1;q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8:out.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');x=a.ic
 print('H',h,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==1:
  for yr,g in x.groupby(x.index.year):print('year',yr,'IC',g.mean(),'ICIR',g.mean()/g.std(ddof=1),'dates',len(g))
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean().mean())
