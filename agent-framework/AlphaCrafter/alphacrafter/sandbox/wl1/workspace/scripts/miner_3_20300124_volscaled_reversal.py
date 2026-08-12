import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({a:get_stock_daily_data(a,days=5000).set_index('date')['close'] for a in U})
rows=[]
for a in U:
 s=px[a].dropna(); r=s.pct_change(); sig=-(s.pct_change(5))/(r.rolling(20,min_periods=15).std()+1e-12)
 for t in sig.index:
  if pd.isna(sig.loc[t]) or t not in px.index: continue
  fut=s.loc[t:].iloc[1:21]
  for h in [1,5,10,20]:
   if len(fut)>=h: rows.append((t,a,sig.loc[t],h,fut.iloc[h-1]/s.loc[t]-1))
out=pd.DataFrame(rows,columns=['date','asset','factor','h','fwd'])
print('rows',len(out),'dates',out.date.nunique(),'assets',out.asset.nunique())
for h,g in out.groupby('h'):
 cs=g.groupby('date').filter(lambda z:len(z)>=8).groupby('date').apply(lambda z:z.factor.corr(z.fwd)).dropna();print('METRIC',h,len(cs),round(cs.mean(),6),round(cs.mean()/cs.std(ddof=1),6),round((cs>0).mean(),4))
for label,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31')]:
 z=out[(out.h==10)&(out.date>=lo)&(out.date<=hi)].groupby('date').apply(lambda q:q.factor.corr(q.fwd)).dropna(); print('REGIME',label,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
out[out.h==10][['date','asset','factor']].to_csv('scripts/miner_3_20300124_volscaled_reversal_signal.csv',index=False)
