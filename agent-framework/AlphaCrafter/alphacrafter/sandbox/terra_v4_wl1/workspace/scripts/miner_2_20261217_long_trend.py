import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in [get_index_daily_data,get_stock_daily_data]:
  try:
   x=fn(s,days=2200)
   if x is not None and len(x)>100:return x[['date','close']]
  except:pass
p=pd.concat([get(s).assign(symbol=s) for s in U]).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=p.pct_change()
# intermediate-horizon trend: 60d return, excluding most recent 5d, divided by 60d vol
for L in [40,60,120]:
 f=(p.shift(5)/p.shift(L+5)-1)/(r.rolling(L).std().shift(5)*np.sqrt(L))
 fr=p.shift(-1)/p-1; a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print(L,len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0))
 print('5d', end=' ')
 fr=p.shift(-5)/p-1;a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
 print('coverage',f.notna().sum(axis=1).mean()/15,'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
