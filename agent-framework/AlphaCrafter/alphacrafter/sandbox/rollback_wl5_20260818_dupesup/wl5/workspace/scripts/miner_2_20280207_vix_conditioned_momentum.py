import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try: d=get_stock_daily_data(s,days=4000)
 except FileNotFoundError: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].rename(s)
v=get_index_daily_data('VIX',days=4000); v=v.set_index('date')['close'].rename('VIX')
P=pd.concat(px.values(),axis=1).sort_index().ffill(); R=P.pct_change();
vix=v.reindex(P.index).ffill(); vz=(vix-vix.rolling(60).mean())/vix.rolling(60).std(); F=R.rolling(20).sum().mul(np.tanh(-vz/2.0),axis=0)
def calc(Y):
 out=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): out.append(c); ns.append(len(z)); dates.append(dt)
 return np.array(out),np.array(ns),dates
for h in [1,5,10,20]:
 a,ns,ds=calc(P.shift(-h)/P-1); print('h',h,'dates',len(a),'meanN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
a,ns,ds=calc(P.shift(-10)/P-1)
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2027','2028')]:
 sub=np.array([c for c,d in zip(a,ds) if lo<=str(d)[:4]<=hi]); print(lo,hi,len(sub),sub.mean() if len(sub) else np.nan,(sub.mean()/sub.std(ddof=1)) if len(sub)>1 else np.nan)
