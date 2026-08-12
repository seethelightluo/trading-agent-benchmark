import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:get_stock_daily_data(a,5000).set_index('date').close for a in U})
R=P.pct_change(); d=get_index_daily_data('DXY',5000).set_index('date').close.reindex(P.index).ffill().pct_change()
# Dollar-beta neutralized medium trend, lagged one completed day
F=pd.DataFrame(index=P.index)
for a in U:
 r=R[a]; cov=r.rolling(60,min_periods=40).cov(d); var=d.rolling(60,min_periods=40).var()
 beta=cov/(var+1e-12); tr=r.rolling(20,min_periods=15).sum(); dr=d.rolling(20,min_periods=15).sum()
 F[a]=(tr-beta*dr).shift(1)
rows=[]
for t in P.index:
 for a in U:
  if pd.notna(F.loc[t,a]):
   base=P.loc[t,a]; fut=P[a].loc[t:].iloc[1:21]
   for h in [1,5,10,20]:
    if len(fut)>=h: rows.append((t,a,F.loc[t,a],h,fut.iloc[h-1]/base-1))
out=pd.DataFrame(rows,columns=['date','asset','factor','h','fwd'])
print('dates',out.date.nunique(),'assets',out.asset.nunique(),'rows',len(out))
for h,g in out.groupby('h'):
 z=g.groupby('date').filter(lambda x:len(x)>=8).groupby('date').apply(lambda x:x.factor.corr(x.fwd)).dropna(); print('METRIC',h,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
x=out[out.h==20]; x[['date','asset','factor']].to_csv('scripts/miner_2_20300207_dxy_neutral_trend_signal.csv',index=False)
for lab,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
 z=x[(x.date>=lo)&(x.date<=hi)].groupby('date').apply(lambda q:q.factor.corr(q.fwd)).dropna(); print('REGIME',lab,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
