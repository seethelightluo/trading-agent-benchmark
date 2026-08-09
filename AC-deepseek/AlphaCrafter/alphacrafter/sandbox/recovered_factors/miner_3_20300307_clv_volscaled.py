import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); r=d.close.pct_change(); rng=(d.high-d.low).replace(0,np.nan)
 clv=(2*d.close-d.high-d.low)/rng; vol=r.rolling(20,min_periods=15).std()
 D[a]=pd.DataFrame({'f':(clv.rolling(10,min_periods=8).mean()/vol),'r':r})
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]));F=pd.DataFrame({a:D[a].f for a in A}).reindex(idx);R=pd.DataFrame({a:D[a].r for a in A}).reindex(idx)
print('dates',idx[0],idx[-1],'assets',len(A),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 fw=R.shift(-h).rolling(h).sum().shift(-(h-1));z=[];ns=[];ds=[]
 for dt in idx:
  x=F.shift(1).loc[dt];y=fw.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());ds.append(dt)
 z=np.array(z);print(h,len(z),round(np.mean(ns),2),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),3))
 for n,m in [('25-27',[2025<=d.year<=2027 for d in ds]),('28-29',[2028<=d.year<=2029 for d in ds]),('latest120',np.arange(len(ds))>=len(ds)-120)]:
  q=z[np.array(m)];print(n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
