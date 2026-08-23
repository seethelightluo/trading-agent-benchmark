import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-11-18');F={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date');d['r']=d.close.pct_change();F[s]=d
# Compression-breakout: recent 5d momentum, scaled by 20d vol and prior 20/60 volatility compression.
rows=[]
for s,d in F.items():
 v20=d.r.rolling(20).std();v60=d.r.rolling(60).std();f=d.close.pct_change(5)/v20*(v60/v20).clip(0.5,3.0);r=d.r.shift(-1)
 for dt in d.index:
  if pd.notna(f.get(dt)) and pd.notna(r.get(dt)):rows.append((dt,s,f.loc[dt],r.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']);n=x.groupby('date').size();ic=x.groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna();print('compression_breakout');print('dates',len(ic),'avg_names',n.mean(),'coverage',n.mean()/15);print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
for h in [5,10,20]:
 z=[]
 for s,d in F.items():
  v20=d.r.rolling(20).std();v60=d.r.rolling(60).std();f=d.close.pct_change(5)/v20*(v60/v20).clip(.5,3);r=d.close.pct_change(h).shift(-h);q=pd.DataFrame({'f':f,'r':r}).dropna();z += [(dt,q.loc[dt].f,q.loc[dt].r) for dt in q.index]
 a=pd.DataFrame(z,columns=['dt','f','r']).groupby('dt').apply(lambda q:q.f.corr(q.r),include_groups=False).dropna();print('h',h,'ICIR',a.mean()/a.std(),'IC',a.mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ic.loc[a:b];print('regime',a,b,len(q),q.mean(),q.mean()/q.std())
print('turnover',x.sort_values(['sym','date']).groupby('sym').f.apply(lambda z:z.rank(pct=True).diff().abs().mean()).mean())
