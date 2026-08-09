import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=d['close'].astype(float)
px=pd.DataFrame(D).sort_index(); ret=px.pct_change()
# candidate: negative 3-session return divided by trailing 20-session realized volatility; all values at t, forward starts t+1
f=-(px.pct_change(3))/(ret.rolling(20,min_periods=15).std()*np.sqrt(3))
# winsorize cross-sectionally is not needed; evaluate
for h in [1,3,5,10,20]:
 fr=px.shift(-h)/px-1
 vals=[]; ns=[]; dates=[]
 for dt in px.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals,index=dates)
 print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'med',round(s.median(),6))
 for label,mask in [('2020-23',s.index<'2024-01-01'),('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',s.index>='2028-01-01'),('last120',s.index>=s.index[-120])]:
  q=s[mask]
  print(' ',label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('date range',px.index.min(),px.index.max())
