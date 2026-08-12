import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
parts=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.sort_values('date').copy(); c=d.close.astype(float); r=c.pct_change()
 parts.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'r5':c.pct_change(5),'r20':c.pct_change(20),'fwd':c.shift(-5)/c-1}))
x=pd.concat(parts,ignore_index=True)
p=x.pivot(index='date',columns='symbol'); med20=p['r20'].median(axis=1)
reg=(med20>0).astype(float)
x['factor_value']=x['r5']*(1-2*reg.loc[x.date].to_numpy())
x['factor_value']=x['factor_value']-x.groupby('date')['factor_value'].transform('median')
a=x[['date','symbol','factor_value','fwd']].rename(columns={'fwd':'forward_return_5d'}).dropna()
a.to_csv('scripts/miner_2_20290208_conditional_reversal_signal.csv',index=False)
z=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.forward_return_5d.nunique()>1:z.append((pd.Timestamp(dt),g.factor_value.corr(g.forward_return_5d,method='spearman')))
ic=pd.Series(dict(z)).sort_index(); print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4),'cutoff',ic.index.max().date())
r=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True); print('turnover_proxy',round((r.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent120',None,None)]:
 q=ic.tail(120) if label=='recent120' else ic.loc[lo:hi]; print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
