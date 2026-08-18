import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data';end=pd.Timestamp('2027-10-18');P={}
for s in U:
 d=pd.read_csv(os.path.join(root,s+'.csv'),parse_dates=['date']).set_index('date').sort_index();P[s]=d.loc[d.index<=end,'close']
p=pd.DataFrame(P).sort_index();r=p.pct_change();f=-(p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0)); y=r.shift(-1);a=[];ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(a);print('dates',len(a),'assets',len(U),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'avgN',np.mean(ns));print('early',a[:len(a)//2].mean(),'late',a[len(a)//2:].mean())
for h in [3,5,10]:
 q=p.shift(-h)/p-1;z=[]
 for dt in f.index:
  x=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 z=np.array(z);print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
