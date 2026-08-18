import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[a]=x
# gap reversal, lagged one day to avoid using same-day open for subsequent close return ambiguity
cl=pd.DataFrame({a:x.close for a,x in D.items()}); op=pd.DataFrame({a:x.open for a,x in D.items()})
gap=op/cl.shift(1)-1
fac=-(gap.rolling(3).mean()).shift(1) # smoothed overnight gap contrarian
rets=cl.pct_change()
for h in [1,5,10,20]:
 fwd=cl.shift(-h)/cl-1
 vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates).dropna(); print('h',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std()*np.sqrt(len(s)),5),'hit',round((s>0).mean(),3),'recent250',round(s.tail(250).mean(),5),'coverage',round(np.mean(ns)/15,4))
# rank turnover and regime
r=fac.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean(),'valid_dates',fac.dropna(how='all').shape[0])
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 q=[]
 for dt in s.index:
  if lo<=str(dt.year)<=hi:
   q.append(s.loc[dt])
 print(lo,round(np.mean(q),5) if q else None,len(q))
