import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change(); v=r.rolling(20,min_periods=10).std().shift(1)
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'r':r,'base':-r/v,'f1':c.shift(-1)/c-1,'f3':c.shift(-3)/c-1}))
a=pd.concat(rows); cs=a.pivot(index='date',columns='symbol',values='r'); disp=cs.std(axis=1).rolling(20,min_periods=10).mean().shift(1); med=disp.rolling(120,min_periods=40).median(); high=(disp>med).astype(float)
a['factor_value']=a['base']*a.date.map(high).fillna(0)
a=a.dropna(subset=['factor_value','f1']);a.to_csv('scripts/miner_2_20290322_dispersion_volreversal_signal.csv',index=False)
for h in [1,3]:
 col='f%d'%h; q=[]
 for dt,g in a.dropna(subset=[col]).groupby('date'):
  if len(g)>=8 and g.factor_value.nunique()>1:q.append(g.factor_value.corr(g[col],method='spearman'))
 x=np.array(q);print(h,len(x),round(x.mean(),6),round(x.mean()/x.std(),6),round((x>0).mean(),4))
print('rows',len(a),'dates',a.date.nunique(),'coverage',len(a)/(a.date.nunique()*15),'turnover',a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True).diff().abs().mean().mean())
