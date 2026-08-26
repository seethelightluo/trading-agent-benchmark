import json
import numpy as np
import pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2033-12-07')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date'); px[s]=d[d.date<=CUT].set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); f=p.pct_change(20)/(r.rolling(40,min_periods=30).std()*np.sqrt(20)+1e-9); y=p.shift(-10)/p-1
ics=[]; ns=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna(); z.columns=['f','y']
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
  ic=z.f.corr(z.y,method='spearman')
  if np.isfinite(ic): ics.append((dt,ic));ns.append(len(z)); rows += [{'date':dt,'symbol':s,'signal':float(z.loc[s,'f']),'forward_return_10d':float(z.loc[s,'y'])} for s in z.index]
i=pd.DataFrame(ics,columns=['date','ic']); m=i.ic.mean(); sd=i.ic.std(ddof=1)
print(json.dumps({'dates':len(i),'instruments':15,'start':str(i.date.min().date()),'end':str(i.date.max().date()),'mean_n':np.mean(ns),'coverage':len(rows)/(len(i)*15),'IC':m,'ICIR':m/sd*np.sqrt(252),'hit_ratio':np.mean(i.ic>0),'turnover':f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()}))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-12-07')]:
 q=i[(i.date>=a)&(i.date<=b)]; print('REGIME',a,b,len(q),q.ic.mean() if len(q) else None)
for h in [5,10,20]:
 yy=p.shift(-h)/p-1; v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('DECAY',h,len(v),np.nanmean(v))
