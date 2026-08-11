import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-10-07'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; D[s]=x[x.close.notna()]
# Volume-confirmed medium momentum: 10d return multiplied by relative 20d average volume, rank cross-section.
for h in [5,10,20]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); v=x.volume.replace(0,np.nan)
  for j in range(30,len(x)-h):
   ret=x.close.iloc[j]/x.close.iloc[j-10]-1
   vr=v.iloc[j-4:j+1].mean()/v.iloc[j-19:j+1].mean()-1
   f=ret*(1+vr); y=x.close.iloc[j+h]/x.close.iloc[j]-1
   if np.isfinite(f) and np.isfinite(y): rows.append((x.index[j],f,y))
 a=pd.DataFrame(rows,columns=['date','f','y']); z=[]; ns=[]; ds=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: z.append(spearmanr(g.f,g.y).statistic); ns.append(len(g)); ds.append(dt)
 z=np.asarray(z); print('horizon',h,'dates',len(z),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
 print('annual',{y:round(z[pd.DatetimeIndex(ds).year==y].mean(),6) for y in sorted(set(pd.DatetimeIndex(ds).year))})
