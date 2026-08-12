import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
mkt=r.sum(axis=1,min_count=8).div(r.notna().sum(axis=1).replace(0,np.nan))
# rolling beta to equal-weight market, computed per asset without future information
mv=mkt.rolling(60,min_periods=40).var()
beta=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r.columns: beta[s]=r[s].rolling(60,min_periods=40).cov(mkt).div(mv)
res=r-beta.mul(mkt,axis=0); f=res.rolling(20,min_periods=15).sum().shift(1)
for h in [1,5,10,20]:
 fr=np.log(p.shift(-h)/p); qs=[];ns=[];ts=[];prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));rk=f.loc[dt].rank(pct=True).reindex(U).fillna(.5)
   if prev is not None:ts.append((rk-prev).abs().mean())
   prev=rk
 q=pd.Series(qs).dropna();print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round(np.mean(q>0),4),'turn',round(np.mean(ts),4))
fr=np.log(p.shift(-10)/p);qs=[];ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
q=pd.Series(qs,index=pd.to_datetime(ds))
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2031-12-31')]:
 z=q.loc[a:b];print('regime',a,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6) if len(z)>1 else None)
f.to_csv('scripts/miner_1_20311113_residual_momentum_signal.csv',index_label='date');print('range',p.index.min(),p.index.max(),'assets',len(p.columns),'coverage',round(f.notna().sum().mean()/len(U),4))
