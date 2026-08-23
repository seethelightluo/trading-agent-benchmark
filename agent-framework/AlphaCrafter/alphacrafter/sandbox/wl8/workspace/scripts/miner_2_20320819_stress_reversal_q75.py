import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-08-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill();r=p.pct_change(); q=v.shift(1).rolling(120,min_periods=60).quantile(.75); f=(-r.shift(1).rolling(5,min_periods=5).sum()).mul((v.shift(1)>q).astype(float),axis=0).rolling(3,min_periods=3).mean(); fr=p.shift(-10)/p-1
def ic(a,b):
 o=a.notna()&b.notna();return spearmanr(a[o],b[o]).statistic if o.sum()>=8 and a[o].nunique()>2 else np.nan
z=[]
for i,d in enumerate(p.index[:-20]):
 if pd.Timestamp('2020-06-01')<=d<=cut:
  x=ic(f.loc[d],fr.loc[d]);
  if pd.notna(x):z.append(x)
z=pd.Series(z);print('dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turn',f.rank(pct=True).diff().abs().mean().mean());print('recent365',z.tail(365).mean(),z.tail(365).mean()/z.tail(365).std(ddof=1))
