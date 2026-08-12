import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=20
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date'); r=d.close.pct_change()
 f=(d.close.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum())).shift(1)
 fr=d.close.shift(-H)/d.close-1
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor_value':f,'forward_return_20d':fr}).dropna())
a=pd.concat(rows,ignore_index=True);out='scripts/miner_1_20290111_path_efficiency20_20d_signal.csv';a.to_csv(out,index=False)
by=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.forward_return_20d.nunique()>1: by.append((pd.Timestamp(dt),g.factor_value.corr(g.forward_return_20d,method='spearman')))
z=pd.Series(dict(by)).sort_index();print('artifact_rows',len(a),'dates',len(z),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2));print('IC20d',round(z.mean(),6),'ICIR20d',round(z.mean()/z.std(),6),'hit',round((z>0).mean(),4));p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(z)*15),4),'cutoff',a.date.max().date())
for label,start,end in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028-12-31','2028-12-31'),('recent252',None,None)]:
 q=z.iloc[-252:] if label=='recent252' else z.loc[start:end];print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
