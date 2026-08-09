import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';macro='../persistent/index_data';cut=pd.Timestamp('2026-11-05')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float);px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index();R=P.pct_change()
d=pd.read_csv(os.path.join(macro,'DXY.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float);mr=d.pct_change().reindex(R.index)
# explicit rolling covariance avoids pandas rolling.cov alignment quirks
n=60; minp=40
xm=R.rolling(n,min_periods=minp).mean(); ym=mr.rolling(n,min_periods=minp).mean()
cov=(R.mul(mr,axis=0)).rolling(n,min_periods=minp).mean()-xm.mul(ym,axis=0)
var=mr.pow(2).rolling(n,min_periods=minp).mean()-ym.pow(2)
f=-cov.div(var,axis=0)
print('factor negative 60d DXY beta; dates',P.index.min(),P.index.max(),'universe',len(U))
for h in [1,3,5,10]:
 out=[]
 for dt in R.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1;q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8:out.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');x=a.ic
 print('H',h,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==1:
  for yr,g in x.groupby(x.index.year):print('year',yr,'IC',g.mean(),'ICIR',g.mean()/g.std(ddof=1),'dates',len(g))
rank=f.rank(axis=1,pct=True);print('coverage',f.notna().sum().sum()/f.size,'turnover',rank.diff().abs().mean().mean())
