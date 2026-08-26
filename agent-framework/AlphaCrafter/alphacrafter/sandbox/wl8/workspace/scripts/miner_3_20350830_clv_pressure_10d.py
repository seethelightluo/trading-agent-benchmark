import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None:
  d=d[['date','high','low','close']].drop_duplicates('date').set_index('date')
  clv=((d.close-d.low)/(d.high-d.low).replace(0,np.nan)-.5)
  fs[s]=clv.rolling(5,min_periods=3).mean()
sig=pd.DataFrame(fs).sort_index().ffill().shift(1)
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None:px[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
px=pd.DataFrame(px).sort_index().ffill(); ics=[];ns=[];tr=[];prev=None
for dt in sig.index:
 z=pd.concat([(-sig.loc[dt]),(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=sig.loc[dt].rank(pct=True).dropna()
 if prev is not None:
  q=x.index.intersection(prev.index)
  if len(q):tr.append(abs(x[q]-prev[q]).mean())
 prev=x
ic=pd.Series(dict(ics));print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(tr))
for h in [1,5,20]:
 a=[];y=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([(-sig.loc[dt]),y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
for n in [365,750,1260]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20350830_clv_pressure_10d_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_3_20350830_clv_pressure_10d_ic.csv',index=False)
