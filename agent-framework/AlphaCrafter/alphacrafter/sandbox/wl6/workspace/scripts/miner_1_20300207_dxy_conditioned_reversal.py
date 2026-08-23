import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 x=get_stock_daily_data(s,days=2700)
 if x is not None and len(x)>120:
  x=x.copy(); x.date=pd.to_datetime(x.date); d[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(d).sort_index().ffill(); r=p.pct_change()
x=pd.read_csv('../persistent/index_data/DXY.csv'); x.date=pd.to_datetime(x.date); dx=x.set_index('date').close.astype(float).reindex(p.index).ffill()
# DXY strength regime: lagged percentile, reverse short-term cross-asset shock only in extremes
pct=dx.rolling(120).rank(pct=True).shift(1)
shock=p.pct_change(5)/r.rolling(20).std().mul(np.sqrt(5)); base=shock
coef=pd.Series(1.,index=p.index); coef[pct>0.8]=-0.5; coef[pct<0.2]=-0.5
f=base.multiply(coef,axis=0); f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 fr=p.shift(-h).div(p).sub(1); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):a.append(c);ns.append(len(z))
 a=np.array(a); return len(a),a.mean(),a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),np.mean(a>0),np.mean(ns)
print('universe',len(d),'dates',len(p),'range',p.index.min(),p.index.max())
for h in [1,5,10,20]:print('H',h,calc(h))
print('coverage',f.notna().sum(axis=1).mean()/len(U))
