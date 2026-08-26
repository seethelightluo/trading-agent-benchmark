import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-31')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index(); r=px.pct_change()
# Trend consistency: medium return weighted by fraction of positive sessions, penalizing choppy paths.
fac=(px/px.shift(20)-1)*(r.gt(0).rolling(20,min_periods=15).mean()-0.5)
def S(a): return (len(a),float(np.mean(a)),float(np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a)))),float(np.mean(a>0)))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; a=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=np.array(a);ds=pd.to_datetime(ds);print(h,S(a),'avgN',round(np.mean(ns),2),'recent252',S(a[-252:]),'2027+',S(a[ds>=pd.Timestamp('2027-01-01')]))
print('coverage',float(fac.notna().mean().mean()),'turnover',float(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2),'dates',len(a))
