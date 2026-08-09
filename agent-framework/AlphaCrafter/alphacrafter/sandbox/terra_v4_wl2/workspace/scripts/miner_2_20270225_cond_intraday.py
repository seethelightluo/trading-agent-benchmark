import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];C={};O={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();C[a]=d.close;O[a]=d.open
p=pd.concat(C,axis=1).sort_index();op=pd.concat(O,axis=1).reindex(p.index); intr=p/op-1
for th in [0.005,0.01,0.02]:
 f=(-(intr)).where(intr.abs()>=th); y=p.shift(-1)/p-1; z=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print(th,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),f.notna().mean().mean())
