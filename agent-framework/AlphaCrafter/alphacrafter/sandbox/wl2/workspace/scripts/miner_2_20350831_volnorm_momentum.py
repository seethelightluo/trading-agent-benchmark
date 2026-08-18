import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}; rows=[]
for s,d in D.items():
 if d is None: continue
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); r=np.log(d.close).diff()
 f=np.log(d.close/d.close.shift(20))/(r.rolling(20).std()*np.sqrt(20)); y=np.log(d.close.shift(-10)/d.close)
 rows.append(pd.DataFrame({'date':d.index,'f':f.values,'y':y.values,'symbol':s}))
x=pd.concat(rows,ignore_index=True); dates=sorted(x.date.dropna().unique()); ics=[]; nobs=[]; active=[]
for dt in dates:
 q=x[x.date==dt].dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: ics.append(q.f.corr(q.y)); nobs.append(len(q)); active.append(dt)
ic=pd.Series(ics); print('candidate=volnorm_momentum_20d horizon=10'); print('dates',len(ic),'avg_instruments',np.mean(nobs),'universe',len(U),'coverage',len(x.dropna())/(len(dates)*len(U))); print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
years=pd.Series(active).astype(str).str[:4]
for label,ys in [('2020-2022',years.isin(['2020','2021','2022'])),('2023-2025',years.isin(['2023','2024','2025'])),('2026-2029',years.isin(['2026','2027','2028','2029'])),('2030-2035',years.isin(['2030','2031','2032','2033','2034','2035']))]:
 a=ic[ys.values]; print(label,len(a),a.mean() if len(a) else np.nan,(a.mean()/a.std(ddof=1)) if len(a)>1 else np.nan)
for h in [1,5,10,20,40]:
 vals=[]
 for s,d in D.items():
  if d is None: continue
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); r=np.log(d.close).diff(); f=np.log(d.close/d.close.shift(20))/(r.rolling(20).std()*np.sqrt(20)); y=np.log(d.close.shift(-h)/d.close); vals.append(pd.DataFrame({'date':d.index,'f':f.values,'y':y.values}))
 a=pd.concat(vals,ignore_index=True).dropna(); cs=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: cs.append(g.f.corr(g.y))
 print('decay',h,np.nanmean(cs),len(cs))
print('last_date',x.date.max())
