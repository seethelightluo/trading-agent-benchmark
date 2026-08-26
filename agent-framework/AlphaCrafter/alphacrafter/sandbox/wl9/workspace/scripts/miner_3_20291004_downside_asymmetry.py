import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Defensive asymmetry: assets with relatively low recent downside volatility versus total volatility rank higher.
down=r.clip(upper=0).rolling(20,min_periods=15).std(); total=r.rolling(60,min_periods=40).std()
sig=-(down/total) # larger (less negative) = lower downside asymmetry risk
print('data',p.index.min().date(),p.index.max().date(),'assets',len(p.columns))
for h in [5,10,20,40]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c); ns.append(len(q)); dates.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(dates)); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for label,st in [('online','2026-07-16'),('recent252','2028-09-20'),('2029','2029-01-01')]:
  z=a[a.index>=st]; print(label,len(z),round(z.mean(),7),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),4))
 if h==10:
  sig.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv('scripts/miner_3_20291004_downside_asymmetry_signal.csv',index=False)
