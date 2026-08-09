import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s, days=4000)
 if d is None or len(d)<40: continue
 d=d.sort_values('date').copy(); d['date']=pd.to_datetime(d.date)
 intr=d.close/d.open-1
 d['f']=-(intr.rolling(3,min_periods=3).mean())
 d['f']=d.f.replace([np.inf,-np.inf],np.nan)
 d['fr']=d.close.shift(-1)/d.close-1
 for _,r in d[['date','f','fr']].dropna().iterrows(): rows.append((r.date,s,r.f,r.fr))
x=pd.DataFrame(rows,columns=['date','symbol','signal','forward_return'])
x.to_csv('../persistent/factor_signals_miner_2_20270225_intraday_reversal3.csv',index=False)
ics=[]; nms=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.forward_return.nunique()>1:
  ics.append(g.signal.rank().corr(g.forward_return.rank())); nms.append(len(g))
a=pd.Series(ics).dropna()
print('dates',len(a),'avg_n',np.mean(nms),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 vals=[]
 for dt,g in x.groupby('date'):
  if lo<=str(dt)[:4]<=hi:
   q=g.dropna()
   if len(q)>=8: vals.append(q.signal.rank().corr(q.forward_return.rank()))
 q=pd.Series(vals).dropna(); print(lo,hi,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
for h in [1,3,5,10]:
 rows2=[]
 for s in U:
  d=get_stock_daily_data(s,days=4000)
  if d is None: continue
  d=d.sort_values('date'); f=-(d.close/d.open-1).rolling(3,min_periods=3).mean(); r=d.close.shift(-h)/d.close-1
  rows2 += list(zip(pd.to_datetime(d.date),f,r))
 y=pd.DataFrame(rows2,columns=['date','f','r']); vv=[]
 for dt,g in y.groupby('date'):
  g=g.dropna()
  if len(g)>=8: vv.append(g.f.rank().corr(g.r.rank()))
 vv=pd.Series(vv).dropna(); print('h',h,'n',len(vv),'ic',vv.mean(),'icir',vv.mean()/vv.std(ddof=1))
print('coverage',x.groupby('date').symbol.nunique().mean()/15)
