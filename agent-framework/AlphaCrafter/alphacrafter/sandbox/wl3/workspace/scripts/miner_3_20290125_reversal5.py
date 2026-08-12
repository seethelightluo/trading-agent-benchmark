import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];z=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date');d['r5']=d.close.pct_change(5);d['fwd']=d.close.shift(-1)/d.close-1;d['symbol']=s;z.append(d[['date','symbol','r5','fwd']])
a=pd.concat(z,ignore_index=True);a['factor_value']=-a.groupby('symbol')['r5'].shift(1);x=a.pivot(index='date',columns='symbol',values='factor_value');a['factor_value']=a['factor_value']-a['date'].map(x.median());a=a.dropna();a.to_csv('scripts/miner_3_20290125_reversal5_signal.csv',index=False)
q=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.fwd.nunique()>1:q.append((pd.Timestamp(dt),g.factor_value.corr(g.fwd,method='spearman')))
ic=pd.Series(dict(q)).sort_index();print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2));print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4));p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('turnover',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4),'cutoff',ic.index.max().date());q=ic.tail(252);print('recent',len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
