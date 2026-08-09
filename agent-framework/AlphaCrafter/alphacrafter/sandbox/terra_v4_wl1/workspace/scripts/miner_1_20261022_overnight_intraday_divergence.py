import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d=d.sort_values('date').reset_index(drop=True); c=d.close.astype(float); o=d.open.astype(float)
 # divergence: recent overnight move versus regular-session move; contrarian to net intraday pressure
 overnight=o/c.shift(1)-1; intraday=c/o-1
 f=(overnight-intraday).rolling(5,min_periods=4).mean()
 y=c.shift(-1)/c-1
 for dt,x,yy in zip(d.date,f,y):
  if np.isfinite(x) and np.isfinite(yy): rows.append((dt,s,x,yy))
x=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8:
  z=g.f.corr(g.y)
  if np.isfinite(z): ics.append((dt,z))
a=pd.Series(dict(ics)); print('dates',len(a),'names_avg',round(x.groupby('date').s.nunique().mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'coverage',round(len(x)/(len(U)*x.date.nunique()),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[(a.index.year>=lo)&(a.index.year<=hi)]; print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
for h in [3,5,10]:
 # build horizon directly per asset and rerun
 rr=[]
 for s,g in x.groupby('s'):
  d=get_stock_daily_data(s,3000).sort_values('date'); c=d.close.astype(float); yy=c.shift(-h)/c-1
  # align factor by date
  f=(d.open.astype(float)/c.shift(1)-1-(c/d.open.astype(float)-1)).rolling(5,min_periods=4).mean()
  rr += list(zip(d.date,f,yy))
 z=pd.DataFrame(rr,columns=['date','f','y']).dropna(); ii=z.groupby('date').apply(lambda g:g.f.corr(g.y)); ii=ii[ii.index.map(lambda d: True)]
 print('horizon',h,'dates',len(ii.dropna()),'IC',round(ii.mean(),6),'ICIR',round(ii.mean()/ii.std(),6))
p=x.pivot_table(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',round(p.diff().abs().mean().mean(),5))
