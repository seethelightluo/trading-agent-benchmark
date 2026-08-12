import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=1900)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
# Candidate: trend acceleration = medium trend relative to long trend, volatility scaled
r=p.pct_change(); vol=r.rolling(20).std()
# use only completed t signal; forward return starts t+1
f=(p.pct_change(20)-p.pct_change(60)/3)/(vol*np.sqrt(20)).replace(0,np.nan)
# cross-sectional rank factor
f=f.rank(axis=1,pct=True)
rows=[]
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h)
 ic=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=pd.Series(ic).dropna(); rows.append((h,len(a),np.mean(ns),a.mean(),a.std(ddof=1),a.mean()/a.std(ddof=1), (a>0).mean()))
print('dates',p.index.min(),p.index.max(),'assets',len(D),'rows',len(p))
print('horizon n avgN IC std ICIR hit')
for x in rows: print('%d %d %.2f %.6f %.6f %.6f %.3f'%x)
# turnover and coverage
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 q=[]
 for dt in f.index:
  if a<=dt.year<=b:
   z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(a,b,len(q),np.nanmean(q))
