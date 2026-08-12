import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(f'{b}/{s}.csv');d.date=pd.to_datetime(d.date);p[s]=d.set_index('date').close
P=pd.DataFrame(p).sort_index().loc[:'2031-02-19']; r=P.pct_change();
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date); v=v.set_index('date').close.reindex(P.index).ffill()
# VIX shock-resilient momentum: relative 20d momentum penalized by volatility and recent VIX shock
m=P.pct_change(20); rel=m.sub(m.mean(axis=1),axis=0); vol=r.rolling(40).std(); shock=v.pct_change(5).clip(-1,1)
f=(rel/(vol+1e-8))*(1-0.5*shock.values[:,None]).shift(1) if False else (rel/(vol+1e-8))*(1-0.5*shock.values[:,None]); f=f.shift(1)
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; a=[];ds=[];nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);nn.append(len(z))
 x=pd.Series(a,index=ds);print(h,len(x),np.mean(nn),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
