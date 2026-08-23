import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);P[a]=d.set_index('date').close.astype(float)
p=pd.concat(P,axis=1).sort_index();v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.reindex(p.index).ffill()
r=-p.shift(1).pct_change(3); lvl=(v.shift(1)-v.shift(1).rolling(252,min_periods=60).median())/(v.shift(1).rolling(252,min_periods=60).std()+1e-12); f=r.mul(1+.5*lvl.clip(lower=0),axis=0);fr=p.pct_change().shift(-1);rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(r),'rows',r.n.sum(),'avg_n',r.n.mean(),'coverage',r.n.sum()/len(r)/15);print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean());
for y in [2020,2021,2022,2023,2024,2025,2026,2027]:
 q=r[r.index.year==y];print(y,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else 0)
