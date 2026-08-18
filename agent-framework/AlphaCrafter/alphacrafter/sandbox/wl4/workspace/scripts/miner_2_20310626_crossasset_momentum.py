import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index();
# raw lagged 20d momentum, demeaned cross-sectionally to reduce market beta
f=p.pct_change(20).shift(1); f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]; ds=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic);ns.append(len(a));ds.append(d)
 z=pd.Series(vals,index=ds); print(h,z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z),np.mean(ns),z.tail(365).mean(),z.tail(730).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
