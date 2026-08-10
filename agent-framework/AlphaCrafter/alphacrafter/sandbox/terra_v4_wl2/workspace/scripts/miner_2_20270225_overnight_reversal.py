import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except:pass
D={s:get(s).set_index('date') for s in U}; c=pd.DataFrame({s:D[s].close for s in U}).sort_index();o=pd.DataFrame({s:D[s].open for s in U}).reindex(c.index)
# Overnight reversal: lagged open / prior close gap, signal opposite gap.
gap=(o/c.shift(1)-1).shift(1);f=-gap;f=f.sub(f.mean(axis=1),axis=0); fr=c.shift(-1)/c-1
def calc(ret):
 a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ret.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
for h in [1,5,10]:print('H',h,calc(c.shift(-h)/c-1))
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[]
 for dt in f.index:
  if str(dt)>=lo and str(dt)<=hi:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q);print('REG',lab,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean());print('turnover',np.mean(f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/(2*len(U))))
