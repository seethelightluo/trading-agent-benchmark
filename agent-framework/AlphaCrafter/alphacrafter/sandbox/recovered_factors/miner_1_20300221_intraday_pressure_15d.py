import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).set_index('date'); rng=(x.high-x.low).replace(0,np.nan); D[a]=((x.close-x.open)/rng).rolling(15,min_periods=12).mean()
F=pd.DataFrame(D); P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in D})
for h in [5,10,15,20]:
 q=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(q).dropna(); print(h,len(q),round(np.mean(ns),2),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),round((q>0).mean(),3))
print('coverage',F.notna().mean().mean(),'turnover10',F.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
