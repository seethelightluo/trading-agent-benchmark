import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-05')
def F(s):
 for fn in [get_index_daily_data,get_stock_daily_data]:
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:F(s) for s in U};D={s:x for s,x in D.items() if x is not None};P=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index(); rows=[]
for s in D:
 # close location within lagged 20-day high-low range; low location predicts rebound
 hi=P[s].rolling(20,min_periods=15).max();lo=P[s].rolling(20,min_periods=15).min(); f=(-(P[s]-lo)/(hi-lo+1e-12)).shift(1)
 rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f,'fr':P[s].shift(-1)/P[s]-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def S(x):
 z=[];ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:z+=[g.f.corr(g.fr,method='spearman')];ns+=[len(g)]
 z=pd.Series(z);return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'dates',q.date.nunique(),'rows',len(q),'avg_n',round(q.groupby('date').size().mean(),2),'coverage',round(len(q)/(q.date.nunique()*len(D)),4))
for h in [1,5,10,20]:
 x=q if h==1 else pd.concat([pd.DataFrame({'date':P.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(P.index),'fr':P[s].shift(-h)/P[s]-1}) for s in D],ignore_index=True).dropna();print('horizon',h,S(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,S(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',round(float(r.diff().abs().mean().mean()),5));q.to_csv('scripts/miner_2_20270505_range_location_signal.csv',index=False)
