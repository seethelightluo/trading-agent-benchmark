import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000); x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill();r=np.log(p).diff(); trend=np.log(p/p.shift(60)); vol=r.rolling(20,min_periods=15).std();
# conditional shock reversal: reverse 5d shock, amplified when longer trend agrees with rebound (mean reversion after extended move)
f=(-np.log(p/p.shift(5))*(1+0.5*(trend.abs()/ (vol*np.sqrt(60))).clip(0,3))).shift(1)
for h in [1,5,10,20]:
 fr=np.log(p.shift(-h)/p);q=[];ns=[];turn=[];prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z)); rr=f.loc[dt].rank(pct=True).reindex(U).fillna(.5)
   if prev is not None:turn.append((rr-prev).abs().mean())
   prev=rr
 q=pd.Series(q).dropna();print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),round((q>0).mean(),4),round(np.mean(turn),4))
