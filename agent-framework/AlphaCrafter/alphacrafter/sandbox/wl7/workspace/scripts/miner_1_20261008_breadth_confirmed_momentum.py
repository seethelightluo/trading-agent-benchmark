import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict()['watch_list']; rows=[]
# Candidate: medium-term momentum confirmed by cross-asset breadth, lagged one day.
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); r=d.close.pct_change()
 mom=r.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()
 rows.append(pd.DataFrame({'f':(mom/vol).shift(1),'r':r.shift(-1),'s':s}))
z=pd.concat(rows).dropna(); wide=z.pivot(columns='s',values='r'); # common dates
# breadth computed from lagged asset returns, avoiding current forward return
allr=[]
for s in U:
 d=get_stock_daily_data(s,3000); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); allr.append(d.close.pct_change().rename(s))
rr=pd.concat(allr,axis=1); breadth=(rr.rolling(5,min_periods=3).mean()>0).mean(axis=1)
# condition: amplify momentum when breadth confirms, reverse/neutralize when breadth weak
z['breadth']=z.index.map(breadth); z['f']=z['f']*(0.5+z['breadth'].clip(0,1))
ics=z.groupby(level=0).apply(lambda q:q.f.corr(q.r) if len(q)>=8 else np.nan).dropna()
print('dates',len(ics),'avg_names',z.groupby(level=0).size().mean(),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'coverage',len(z)/(len(ics)*len(U)))
print('regimes',[(int(y),round(x.mean(),4),round(x.mean()/x.std(ddof=1),3),len(x)) for y,x in ics.groupby(ics.index.year)])
print('turnover',z.groupby('s').f.apply(lambda x:(x.diff().abs()>x.rolling(20).std()).mean()).mean())
for h in [5,10]:
 # recompute approximate forward compounded from raw prices
 print('horizon',h)
