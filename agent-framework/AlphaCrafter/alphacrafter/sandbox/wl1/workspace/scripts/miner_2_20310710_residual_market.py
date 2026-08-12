import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff(); m=np.log(p/p.shift(20)); market=r.mean(axis=1).rolling(20).sum(); beta=r.rolling(60).cov(r.mean(axis=1)).div(r.mean(axis=1).rolling(60).var(),axis=0)
sig=m-beta*market
rank=sig.rank(axis=1,pct=True)
for h in [1,5,10,20]:
 f=np.log(p.shift(-h)/p); z=[];ns=[]
 for dt in rank.index:
  a=pd.concat([rank.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a))
 z=np.array(z);print(h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(sig.notna().sum().sum()/sig.size,4),'assets',len(p.columns))
rank.index=rank.index.strftime('%Y-%m-%d');rank.to_csv('scripts/miner_2_20310710_residual_market_signal.csv')
