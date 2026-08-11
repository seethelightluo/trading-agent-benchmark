import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2800)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=2800)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Slow downside-risk trend: medium trend divided by downside semideviation, with a mild breadth regime gate.
ret60=r.rolling(60,min_periods=60).sum(); dn=(-r).clip(lower=0).rolling(60,min_periods=60).std()*np.sqrt(60)
breadth=(r.rolling(20,min_periods=20).sum()>0).mean(axis=1)
f=ret60.div(dn.replace(0,np.nan)).mul((0.75+0.5*breadth),axis=0).shift(1)
def ev(Y,sl=slice(None)):
 a=[]; ns=[]
 for dt in f.loc[sl].index:
  z=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q);ns.append(len(z))
 a=np.asarray(a)
 return len(a),round(float(np.mean(ns)),2),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)),6),round(float((a>0).mean()),4)
for h in [1,3,5,10]: print('h',h,ev(np.log(p).shift(-h)-np.log(p)))
print('coverage',round(float(f.notna().sum(axis=1).mean()/15),4),'turnover',round(float(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean()),5),'dates',len(p),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]: print(n,ev(np.log(p).shift(-1)-np.log(p),s))
