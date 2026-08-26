import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index().ffill();r=p.pct_change(); fr=p.pct_change(10).shift(-10)
for w in [5,10,15,30,40,60]:
 f=(-r.rolling(w).std()).shift(1); a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print(w,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
