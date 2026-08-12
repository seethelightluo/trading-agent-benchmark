import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=5000)
 return d if d is not None and len(d)>=100 else get_index_daily_data(s,days=5000)
D={s:fetch(s) for s in U}; c=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill(); r=np.log(c).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(c.index).ffill(); vm=np.log(v).diff()
b=r.rolling(60,min_periods=40).cov(vm).div(vm.rolling(60,min_periods=40).var(),axis=0); sh=(v/v.rolling(60,min_periods=40).median()-1).clip(-2,2)
f=(b.rolling(5).mean().mul(sh,axis=0)+r.rolling(20).sum()/r.rolling(20).std()).shift(1)
for h in [10,20]:
 y=c.shift(-h)/c-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z))
 a=np.array(a);print('h',h,'dates',len(a),'N',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for st in ['2027-01-01','2028-01-01','2028-06-01']:
 y=c.shift(-10)/c-1;a=[]
 for dt in f.index[f.index>=st]:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 a=np.array(a);print(st,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
print('last',c.index.max().date())
