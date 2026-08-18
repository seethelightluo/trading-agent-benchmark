import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3200);D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();v10=r.rolling(10).std();v40=r.rolling(40).std();v60=r.rolling(60).std();fr=p.shift(-10)/p-1
for h in [5,10,20]:
 R=p.pct_change(h);res=R-R.median(axis=1).values[:,None]
 for typ,g in [('lowvol',(v10/v60).clip(.33,2)),('binary',(v10<v60).astype(float)),('none',1)]:
  f=(-res/v40*g).shift(1);o=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:o.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  o=np.array(o);print(h,typ,len(o),round(np.nanmean(o),6),round(np.nanmean(o)/(np.nanstd(o,ddof=1)/np.sqrt(np.sum(np.isfinite(o)))),4),round(np.mean(o>0),4))
