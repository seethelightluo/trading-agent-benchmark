import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].sort_index()
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); dates=[]; out={5:[],10:[],20:[]}; ns=[]
for i in range(80,len(P)-20):
 r20=P.iloc[i-1]/P.iloc[i-21]-1; r60=P.iloc[i-1]/P.iloc[i-61]-1
 down=R.iloc[:i].tail(20).clip(upper=0).std().replace(0,np.nan)
 sig=(r20-r60/3)/(down*np.sqrt(20)+0.01)
 dates.append(P.index[i]); ns.append(sig.notna().sum())
 for h in out:
  fr=P.iloc[i+h-1]/P.iloc[i-1]-1; z=pd.concat([sig,fr],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); out[h].append(c if np.isfinite(c) else np.nan)
for h in out:
 a=pd.Series(out[h]).dropna(); print('H',h,'dates',len(a),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1),7),'hit',round((a>0).mean(),4))
# Date alignment is exact because every horizon loop has same eligible dates in this construction
for label,st in [('online','2026-07-16'),('recent252','2028-09-20'),('2029','2029-01-01'),('2027_28','2027-01-01')]:
 mask=pd.to_datetime(dates)>=st
 a=pd.Series(out[10])[mask].dropna()
 print(label,'dates',len(a),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1),7) if len(a)>1 else np.nan)
print('data',P.index.min().date(),P.index.max().date(),'assets',len(P.columns),'mean valid assets',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4))
