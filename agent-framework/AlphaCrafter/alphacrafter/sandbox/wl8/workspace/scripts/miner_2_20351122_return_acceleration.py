import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for s in U:
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)<100:d=get_index_daily_data(s,6000)
 if d is not None:p[s]=d.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(p).sort_index().ffill(); r=px.pct_change()
# Acceleration: recent 20-session return versus prior 60-session average, lagged one day.
sig=(px.pct_change(20)-px.pct_change(60)/3).shift(1)
ics=[]; ns=[]; turns=[]; prev=None
for dt in sig.index:
 y=(px.shift(-10)/px-1).loc[dt]; z=pd.concat([sig.loc[dt],y],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):ics.append((dt,c));ns.append(len(z))
 x=sig.loc[dt].rank(pct=True).dropna()
 if prev is not None:
  q=x.index.intersection(prev.index)
  if len(q):turns.append(abs(x[q]-prev[q]).mean())
 prev=x
ic=pd.Series(dict(ics));print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turns))
for h in [1,5,20]:
 a=[]; y=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c)
 print('decay',h,np.mean(a),len(a))
for n in [365,750,1260]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20351122_return_acceleration_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_2_20351122_return_acceleration_ic.csv',index=False)
