import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data';p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index();r=p.pct_change()
for short,long in [(5,20),(5,40),(10,60),(20,60),(10,80)]:
 v1=r.rolling(short,min_periods=max(4,short//2)).std();v2=r.rolling(long,min_periods=max(15,long//2)).std();f=(v1/v2-1).shift(1);y=p.shift(-5)/p-1;z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.asarray(z); print(short,long,'dates',len(z),'N',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'recent',round(z[-124:].mean(),5),round(z[-124:].mean()/z[-124:].std(ddof=1),5),'cov',round(f.notna().stack().mean(),4))
