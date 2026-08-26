import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d.set_index('date')['close']
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Contrarian short-horizon return, normalized by trailing volatility; all inputs through t
vol=r.rolling(20,min_periods=15).std()+1e-8
fac=-(px/px.shift(3)-1)/vol

def stats(a):
 a=np.asarray(a,float); a=a[np.isfinite(a)]
 return (len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(len(a))),float((a>0).mean()))
allres={}
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]; ds=[]; ns=[]
 for i,dt in enumerate(px.index):
  z=pd.concat([fac.iloc[i],fw.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(z))
 a=np.array(vals); ds=pd.to_datetime(ds); allres[h]=stats(a)
 print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'full',stats(a),'recent252',stats(a[-252:]),'online',stats(a[ds>=pd.Timestamp('2026-07-16')]))
rank=fac.rank(axis=1,pct=True)
print('coverage',round(float(fac.notna().sum(axis=1).mean()/15),4),'turnover5',round(float((rank-rank.shift(5)).abs().mean(axis=1).mean()),4),'period',px.index.min(),px.index.max())
