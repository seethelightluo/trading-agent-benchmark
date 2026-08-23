import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d.set_index('date').sort_index()
rows=[]
for s,d in F.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 f=((r>0).rolling(40).mean()-(r<0).rolling(40).mean())*(c.pct_change(40).abs()+.01); f=f.shift(1)
 for h in [1,5,10]:
  y=c.shift(-h)/c-1
  rows += [(dt,s,h,a,b) for dt,a,b in zip(c.index,f,y) if pd.notna(a) and pd.notna(b)]
x=pd.DataFrame(rows,columns=['date','asset','h','f','y'])
for h in [1,5,10]:
 z=x[x.h==h]; vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((dt,g.f.corr(g.y,method='spearman')))
 a=pd.Series(dict(vals)).dropna(); print('horizon',h,'dates',len(a),'avg_n',round(z.groupby('date').size().mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
 if h==1:
  for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-07-07')]:
   q=a[(a.index>=lo)&(a.index<=hi)]; print('regime',label,len(q),round(q.mean(),6))
x[x.h==1].to_csv('scripts/miner_3_20270707_persistence_signal.csv',index=False)
print('assets',len(F),'coverage',len(x[x.h==1])/sum(len(d) for d in F.values()))
