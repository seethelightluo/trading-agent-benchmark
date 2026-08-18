import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data';end=pd.Timestamp('2027-10-18');C={}
for s in U:
 d=pd.read_csv(os.path.join(root,s+'.csv'));d.date=pd.to_datetime(d.date);d=d[(d.date>='2020-01-01')&(d.date<=end)].sort_values('date');C[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index();r=p.pct_change();f=-r.rolling(20,min_periods=15).std();out={}
for h in [1,5,10]:
 y=p.shift(-h)/p-1;ic=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ic);out[h]=(len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns))
print(out);print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
