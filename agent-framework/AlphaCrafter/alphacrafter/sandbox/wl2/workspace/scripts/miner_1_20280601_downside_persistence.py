import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Defensive persistence: medium momentum divided by downside semideviation, with broad-market breadth attenuation.
down=(-r).clip(lower=0).rolling(30,min_periods=20).std(); m20=r.rolling(20,min_periods=20).sum(); breadth=(m20>0).mean(axis=1)
f=m20.div(down.replace(0,np.nan)).mul((0.6+0.8*breadth),axis=0).shift(1)
def ev(h,sl=slice(None)):
 y=np.log(p).shift(-h)-np.log(p);a=[];ns=[]
 for dt in f.loc[sl].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 x=np.asarray(a);return len(x),round(np.mean(ns),2),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4)
for h in [1,3,5,10]:print('h',h,ev(h))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5),'dates',len(p),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:print(n,ev(10,s))
