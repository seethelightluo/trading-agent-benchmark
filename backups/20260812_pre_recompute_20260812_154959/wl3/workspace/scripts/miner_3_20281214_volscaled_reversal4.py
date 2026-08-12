import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date'); ret=d.close.pct_change();vol=ret.rolling(20,min_periods=15).std()
 f=-(d.close.pct_change(4)/(vol*np.sqrt(4))).replace([np.inf,-np.inf],np.nan)
 r=d.close.shift(-1)/d.close-1
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor_value':f,'forward_return_1d':r}).dropna())
a=pd.concat(rows,ignore_index=True);a.to_csv('scripts/miner_3_20281214_volscaled_reversal4_signal.csv',index=False)
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.forward_return_1d.nunique()>1:ics.append(g.factor_value.corr(g.forward_return_1d,method='spearman'))
ic=pd.Series(ics);print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2));print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4));p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4));print('cutoff',a.date.max().date())
for label,sl in [('2020-22',slice('2020','2022-12-31')),('2023-25',slice('2023','2025-12-31')),('2026-27',slice('2026','2027-12-31')),('2028',slice('2028','2028-12-31'))]:
 q=pd.Series(ics,index=sorted(a.date.unique())[:len(ics)]) if False else None
 # recompute by date for regime
 z=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.factor_value.nunique()>1 and g.forward_return_1d.nunique()>1:z.append((pd.Timestamp(dt),g.factor_value.corr(g.forward_return_1d,method='spearman')))
 z=pd.Series(dict(z)).sort_index().loc[sl];print(label,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(),6))
