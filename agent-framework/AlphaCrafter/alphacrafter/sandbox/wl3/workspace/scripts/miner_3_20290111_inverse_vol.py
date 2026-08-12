import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date');r=d.close.pct_change()
 # lagged inverse volatility, cross-sectionally demeaned, no future information
 v=r.rolling(20,min_periods=15).std().shift(1)
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor_value':(1/v).replace([np.inf,-np.inf],np.nan),'forward_return_1d':d.close.shift(-1)/d.close-1}))
a=pd.concat(rows,ignore_index=True).dropna(); x=a.pivot(index='date',columns='symbol',values='factor_value');a['factor_value']=a['factor_value']-a['date'].map(x.mean(axis=1));a.to_csv('scripts/miner_3_20290111_inverse_vol_signal.csv',index=False)
z=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.forward_return_1d.nunique()>1:z.append((pd.Timestamp(dt),g.factor_value.corr(g.forward_return_1d,method='spearman')))
ic=pd.Series(dict(z)).sort_index();print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2));print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4));p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4),'cutoff',ic.index.max().date())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent252',None,None)]:
 q=ic if label!='recent252' else ic.tail(252)
 if label!='recent252':q=q.loc[lo:hi]
 print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
