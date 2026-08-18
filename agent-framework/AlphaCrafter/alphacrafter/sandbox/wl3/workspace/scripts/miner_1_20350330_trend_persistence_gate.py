import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=6000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float).sort_index()
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change(); ret20=px/px.shift(20)-1; ret60=px/px.shift(60)-1; vol20=r.rolling(20).std()*np.sqrt(252)
f=(ret20-ret60/3)/vol20; f=f.where(ret60>0,f*.25); fr=px.shift(-10)/px-1
rows=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def met(q):
 x=a if q is None else a.loc[a.index>=a.index[-1]-pd.Timedelta(days=q)]; return len(x),x.n.mean(),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()
print('span',px.index.min().date(),px.index.max().date(),'dates',len(a),'assets',len(D))
for q in [None,365,730]: print('window',q,met(q))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; rr=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(f.loc[dt].corr(yy.loc[dt],method='spearman'))
 print('decay',h,np.nanmean(rr),len(rr))
ranks=f.rank(axis=1,pct=True); print('turnover',((ranks-ranks.shift()).abs().mean(axis=1)).dropna().mean(),'coverage',a.n.mean()/len(U))
for i,ix in enumerate(np.array_split(np.arange(len(a)),4),1):
 b=a.iloc[ix]; print('block',i,len(b),b.ic.mean(),b.ic.mean()/b.ic.std(ddof=1))
