import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-07-20')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
def ic(a,b):
 ok=a.notna()&b.notna()
 return spearmanr(a[ok],b[ok]).statistic if ok.sum()>=8 else np.nan
for sm in [1,2,3,5,10]:
 f=(p.shift(1)/p.shift(21)-1)/vol.shift(1); f=f.sub(f.median(axis=1),axis=0).rolling(sm,min_periods=sm).mean(); z=[]
 for i,d in enumerate(p.index):
  if d<pd.Timestamp('2020-03-01') or d>cut or i+10>=len(p):continue
  q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d]);
  if pd.notna(q):z.append(q)
 z=pd.Series(z);print('smooth',sm,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'recent365',z.tail(365).mean(),'coverage',f.loc[(p.index>=pd.Timestamp('2020-03-01'))&(p.index<=cut)].notna().mean().mean(),'turn',f.rank(pct=True).diff().abs().mean().mean())
