import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=3200); x=x.set_index(pd.to_datetime(x.date)).sort_index(); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); R=p.pct_change(10); res=R-R.median(axis=1).values[:,None]; v10=r.rolling(10).std();v40=r.rolling(40).std();v60=r.rolling(60).std(); c=(v60/v10).clip(.5,3)
fr=p.shift(-10)/p-1
for name,g in [('compressed',(c-1).clip(0,2)),('rankcomp',c.rank(axis=1,pct=True)),('lowvol',(1/c).clip(.33,2))]:
 f=(-res/v40*g).shift(1); out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 out=np.array(out); print(name,'dates',len(out),'avgN',len(U),'IC',np.nanmean(out),'ICIR',np.nanmean(out)/(np.nanstd(out,ddof=1)/np.sqrt(len(out))),'hit',np.mean(out>0))
 for n in [365,730,1095]:
  q=out[-n:];print(n,round(np.nanmean(q),5),round(np.nanmean(q)/(np.nanstd(q,ddof=1)/np.sqrt(len(q))),4))
