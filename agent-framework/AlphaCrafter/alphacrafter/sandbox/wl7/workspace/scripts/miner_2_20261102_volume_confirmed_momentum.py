import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<40: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date'); c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); vol=r.rolling(20).std()
 f=(-c.pct_change(3)/(vol*np.sqrt(3))).where(vol>1e-8)
 # positive volume surprise confirms medium-term trend, with range fallback if volume unavailable
 v=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan); lv=np.log(v); vz=(lv-lv.rolling(20,min_periods=15).mean())/(lv.rolling(20,min_periods=15).std()+1e-9)
 f=(c.pct_change(10)/(vol*np.sqrt(10))*np.tanh(vz.clip(-3,3)/2)).shift(1)
 z=pd.DataFrame({'date':d.date,'s':s,'f':f,'r':r})
 for h in [1,5,10,20]: z['fr'+str(h)]=c.shift(-h)/c-1
 rows.append(z)
x=pd.concat(rows,ignore_index=True); ics=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fr1'])
 if len(g)>=8: ics.append(g.f.corr(g.fr1,method='spearman'))
a=np.array(ics); ranks=x.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(a),'avg_names',x.dropna(subset=['f','fr1']).groupby('date').s.count().mean(),'coverage',x.dropna(subset=['f','fr1']).groupby('date').s.count().mean()/15,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turnover',turn)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 z=[]
 for dt,g in x[(x.date>=lo)&(x.date<=hi)].groupby('date'):
  g=g.dropna(subset=['f','fr1'])
  if len(g)>=8:z.append(g.f.corr(g.fr1,method='spearman'))
 z=np.array(z); print(lo,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1))
for h in [5,10,20]:
 z=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['f','fr'+str(h)])
  if len(g)>=8:z.append(g.f.corr(g['fr'+str(h)],method='spearman'))
 print('decay',h,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1))
