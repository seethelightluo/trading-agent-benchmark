import os, json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in U:
    try: d=get_stock_daily_data(s,5000)
    except Exception: d=None
    if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(prices).sort_index().ffill()
ret20=px.pct_change(20); ret60=px.pct_change(60); f=ret20-ret60/3
fr=px.shift(-10)/px-1
rows=[]; ics=[]
for dt in px.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna(); z.columns=['f','y']
 if len(z)>=8:
  ic=z.f.corr(z.y,method='spearman')
  if np.isfinite(ic):
   ics.append((dt,ic,len(z)))
   for a,v in z.f.items(): rows.append({'date':dt,'symbol':a,'signal':float(v),'forward_return_10d':float(z.loc[a,'y'])})
i=pd.DataFrame(ics,columns=['date','ic','n']); meanic=i.ic.mean(); icir=meanic/i.ic.std(ddof=1)*np.sqrt(252)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print(json.dumps({'dates':len(i),'start':str(i.date.min().date()),'end':str(i.date.max().date()),'mean_n':i.n.mean(),'coverage':len(rows)/(len(i)*15),'IC':meanic,'ICIR':icir,'hit':(i.ic>0).mean(),'turnover':z.diff().abs().mean(axis=1).mean()},default=str))
for a,b in [('2026-08-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-12-31')]:
 q=i[(i.date>=a)&(i.date<=b)]
 if len(q): print('REGIME',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1; vv=[]
 for dt in px.index:
  q=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(q)>=8: vv.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('DECAY',h,np.nanmean(vv),len(vv))
pd.DataFrame(rows).to_csv('scripts/miner_1_20331124_trend_acceleration_signal.csv',index=False)
