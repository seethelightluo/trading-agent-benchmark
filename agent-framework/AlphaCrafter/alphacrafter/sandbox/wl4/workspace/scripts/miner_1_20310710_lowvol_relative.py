import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();r=p.pct_change();
# Cross-sectional low volatility, normalized by contemporaneous universe median to isolate relative risk.
f=(-(r.rolling(30).std()/(r.rolling(120).std()+1e-12))).shift(1)
for h in [5,10,20]:
 fw=p.shift(-h)/p-1; a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(z))
 a=pd.Series(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recent252',round(a.tail(252).mean(),6),'recentIR',round(a.tail(252).mean()/a.tail(252).std(ddof=1),6))
print('assets',len(D),'dates',len(p),'coverage',round(f.notna().sum(axis=1).mean()/15,4))
