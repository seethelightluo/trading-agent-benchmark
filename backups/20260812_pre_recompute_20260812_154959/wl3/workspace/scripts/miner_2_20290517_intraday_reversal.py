import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change(); rv=r.rolling(20,min_periods=10).std()
 # Volatility-scaled one-day reversal, using information through t-1.
 f=-r.shift(1)/rv.shift(1)
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'x':f,'ret':c.shift(-1)/c-1}))
a=pd.concat(rows).dropna(); out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.x.nunique()>1 and g.ret.nunique()>1:out.append((dt,g.x.corr(g.ret,method='spearman')))
ic=pd.Series(dict(out)).sort_index(); a.rename(columns={'x':'factor_value','ret':'fwd'}).to_csv('scripts/miner_2_20290517_intraday_reversal_signal.csv',index=False)
print('rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4)); print('coverage',round(len(a)/(len(ic)*15),4),'turnover',round((a.pivot(index='date',columns='symbol',values='x').rank(axis=1,pct=True).diff().abs()>0.2).mean(axis=1).mean(),4),'cutoff',ic.index.max())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent252',None,None)]:
 q=ic.tail(252) if label=='recent252' else ic.loc[lo:hi];print(label,len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
for h in [1,3,5,10]:
 z=a.copy(); # reconstruct forward horizon from per-symbol prices unavailable here; use aligned original frames
 vals=[]
 for s in U:
  d=get_stock_daily_data(s,4000)
  if d is None or len(d)==0:d=get_index_daily_data(s,4000)
  if d is None or len(d)==0:continue
  d=d.sort_values('date');c=d.close.astype(float);r=c.pct_change();rv=r.rolling(20,min_periods=10).std(); f=-r.shift(1)/rv.shift(1); vals.append(pd.DataFrame({'date':pd.to_datetime(d.date),'x':f,'y':c.shift(-h)/c-1}))
 z=pd.concat(vals); ii=[]
 for dt,g in z.groupby('date'):
  g=g.dropna()
  if len(g)>=8:ii.append(g.x.corr(g.y,method='spearman'))
 print('decay',h,round(np.nanmean(ii),6),len(ii),round(np.nanmean(ii)/np.nanstd(ii),6))
