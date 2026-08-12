import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fdata(s,look,h):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index();v=d.close.pct_change().rolling(20,min_periods=15).std()
 f=-(d.close.pct_change(look)/(v*np.sqrt(look))).replace([np.inf,-np.inf],np.nan);r=d.close.shift(-h)/d.close-1
 return pd.DataFrame({'f':f,'r':r}).dropna().reset_index()
def ic(q):
 z={}
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:z[dt]=g.f.corr(g.r,method='spearman')
 return pd.Series(z).dropna()
for look in [2,3,4,7,10]:
 for h in [1,3,5]:
  qs=[fdata(s,look,h) for s in U];qs=[q for q in qs if q is not None];a=ic(pd.concat(qs));print('look',look,'h',h,'n',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),5),'cov',round(sum(len(q) for q in qs)/(a.size*15),4))
