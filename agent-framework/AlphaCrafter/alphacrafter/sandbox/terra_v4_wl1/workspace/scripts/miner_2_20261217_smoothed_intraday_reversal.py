import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; fs={}
for s in assets:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<40: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date')
 intr=d.close/d.open-1.0
 # Smooth completed intraday reversal over 3 sessions; lagged factor predicts next close return
 fs[s]=pd.DataFrame({'f':-intr.rolling(3,min_periods=3).mean(),'r':d.close.shift(-1)/d.close-1.0})
F=pd.concat({s:x.f for s,x in fs.items()},axis=1); R=pd.concat({s:x.r for s,x in fs.items()},axis=1)
ics=[]; counts=[]; turns=[]; prev=None
for dt in F.index:
 z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
 x=F.loc[dt].dropna()
 if len(x)>=8:
  q=x.rank(pct=True)
  if prev is not None:
   a=pd.concat([prev,q],axis=1).dropna(); turns.append((a.iloc[:,0]-a.iloc[:,1]).abs().mean())
  prev=q
v=pd.Series(ics).dropna(); print('dates',len(v),'avg_n',np.mean(counts),'coverage',np.mean(counts)/len(assets),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0),'turnover',np.mean(turns))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=[]
 for dt in F.loc[a:b].index:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
# same factor signal artifact, index dates and columns are recoverable
F.to_csv('scripts/miner_2_20261217_smoothed_intraday_reversal_signal.csv')
