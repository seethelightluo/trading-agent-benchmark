import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.sort_values('date').copy(); r=d.close.pct_change()
 # short-horizon reversal scaled by trailing volatility, all lagged through prior close
 f=-(d.close.pct_change(3))/(r.rolling(20).std()*np.sqrt(3))
 rows += [dict(date=pd.Timestamp(dt),symbol=s,factor_value=v,fwd=fwd) for dt,v,fwd in zip(d.date,f,d.close.shift(-1)/d.close-1)]
a=pd.DataFrame(rows).dropna(); a.to_csv('scripts/miner_1_20290222_volscaled_reversal3_signal.csv',index=False)
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.fwd.nunique()>1: ics.append((pd.Timestamp(dt),g.factor_value.corr(g.fwd,method='spearman')))
ic=pd.Series(dict(ics)).sort_index(); print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4)); p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True); print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4),'cutoff',ic.index.max().date())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent252',None,None)]:
 q=ic.tail(252) if label=='recent252' else ic.loc[lo:hi]
 print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
for h in [3,5,10]:
 # evaluate forward cumulative h-day return from current close
 z=[]
 for s in U:
  d=get_stock_daily_data(s,4000)
  if d is None or len(d)==0:d=get_index_daily_data(s,4000)
  if d is not None and len(d):
   d=d.sort_values('date'); r=d.close.pct_change(); f=-(d.close.pct_change(3))/(r.rolling(20).std()*np.sqrt(3)); fr=d.close.shift(-h)/d.close-1
   z += [pd.DataFrame({'date':pd.to_datetime(d.date),'f':f,'fr':fr,'symbol':s})]
 z=pd.concat(z).dropna(); ii=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8: ii.append(g.f.corr(g.fr,method='spearman'))
 print('horizon',h,'IC',round(np.nanmean(ii),6),'n',len(ii))
