import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-09-09')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change(); common=r.mean(axis=1);res=r.sub(common,axis=0); rv=res.rolling(20,min_periods=20).std().shift(1)
def ic(a,b):
 ok=a.notna()&b.notna()
 return spearmanr(a[ok],b[ok]).statistic if ok.sum()>=8 else np.nan
for name,f in [('raw3',-res.rolling(3,min_periods=3).sum().shift(1)),('scaled3',-res.rolling(3,min_periods=3).sum().shift(1)/rv),('clip3',(-res.rolling(3,min_periods=3).sum().shift(1)/rv).clip(-2,2))]:
 z=[]
 for i,d in enumerate(p.index[:-20]):
  if d<pd.Timestamp('2020-04-01') or p.index[i+10]>cut:continue
  q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d]);
  if pd.notna(q):z.append(q)
 z=pd.Series(z);print(name,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),f.notna().mean().mean())
