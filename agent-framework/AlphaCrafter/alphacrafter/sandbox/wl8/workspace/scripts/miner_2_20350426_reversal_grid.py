import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data'
p=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();p=p.loc[:'2035-04-26'];r=np.log(p).diff();fr=p.shift(-10)/p-1
for h,w in [(3,10),(5,10),(5,30),(10,20),(10,40),(15,40),(20,60)]:
 f=-r.shift(1).rolling(h).sum()/(r.shift(1).rolling(w).std()+1e-8);a=[]; ns=[]
 for d in p.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):a.append(v);ns.append(len(z))
 a=np.array(a);print(h,w,'dates',len(a),'N',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
