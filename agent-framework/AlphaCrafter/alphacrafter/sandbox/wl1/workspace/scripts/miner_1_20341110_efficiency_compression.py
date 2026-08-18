import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
net=np.log(p/p.shift(20)); vol=r.rolling(20).std()*np.sqrt(252); eff=net.abs()/r.abs().rolling(20).sum().replace(0,np.nan)
v10=r.rolling(10).std(); med=v10.rolling(60).median(); raw=(net/vol)*eff
f=raw.where(v10<med, raw*0.5).shift(1)
out=[]
for h in [5,10,20,40]:
 fr=np.log(p.shift(-h)/p); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals).dropna(); out.append((h,len(q),np.mean(ns),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if q.std(ddof=1)>0 else np.nan,(q>0).mean()))
print('dates',len(p),'instruments',len(p.columns),'range',p.index.min(),p.index.max()); print('h,n,avgN,IC,ICIR,hit')
for x in out: print('%s %d %.2f %.6f %.6f %.6f %.4f'%x)
h=10; fr=np.log(p.shift(-h)/p); vals=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
q=pd.Series(vals,index=dates).dropna(); print('daily10 mean/std/sqrt252',q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(252),'coverage',f.notna().sum(axis=1).mean()/15)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b]; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20341110_efficiency_compression_signal.csv',index=False)
