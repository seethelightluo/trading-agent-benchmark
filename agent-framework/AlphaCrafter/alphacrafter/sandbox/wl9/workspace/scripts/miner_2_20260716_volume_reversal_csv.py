import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
px={}; vv={}
for a in A:
 z=pd.read_csv(root+'/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();px[a]=z.close.astype(float);vv[a]=z.volume.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2026-07-15']; v=pd.concat(vv,axis=1).reindex(p.index).ffill(); r=p.pct_change(fill_method=None)
vs=v/v.rolling(20,min_periods=10).median()-1
f=-p.pct_change(3,fill_method=None)*vs.clip(-2,2)
for h in [1,5,10]:
 obs=[];ns=[]; dates=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   obs.append(spearmanr(q.f,q.y).statistic);ns.append(len(q));dates.append(p.index[i])
 x=pd.Series(obs,index=dates);print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round((x>0).mean(),4),'coverage',round(np.mean(ns)/15,4),'recent250',round(x.tail(250).mean(),5))
print('period',p.index.min().date(),p.index.max().date())
