import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=2800)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=2800)
 if d is not None:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff()
# Contrarian shock after controlling for each asset's medium trend: recent 3d loss is favored only when 30d trend remains positive.
shock=r.rolling(3,min_periods=3).sum(); trend=r.rolling(30,min_periods=30).sum(); vol=r.rolling(20,min_periods=15).std()
f=(-shock/(vol*np.sqrt(3)+1e-8)*(0.5+0.5*(trend>0))).shift(1)
def ev(Y,sl=slice(None)):
 a=[];ns=[]
 for dt in f.loc[sl].index:
  z=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
for h in [1,3,5,10]:print('h',h,ev(np.log(p).shift(-h)-np.log(p)))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),5),'dates',len(p),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:print(n,ev(np.log(p).shift(-1)-np.log(p),s))
