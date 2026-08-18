import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); dc='date' if 'date' in d else d.columns[0]; P[s]=pd.Series(d.close.astype(float).values,index=pd.to_datetime(d[dc]));
p=pd.DataFrame(P).sort_index().loc[:'2033-04-29'].ffill(); ret=p.pct_change(); vol=ret.rolling(20).std();
# liquidity-confirmed short-term reversal: fade 5d move, stronger after unusual volume, lagged
v={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); dc='date' if 'date' in d else d.columns[0]; v[s]=pd.Series(d.volume.astype(float).values,index=pd.to_datetime(d[dc]))
volu=pd.DataFrame(v).sort_index().replace(0,np.nan).ffill(); surprise=volu/volu.rolling(40).median()-1
f=-(p.pct_change(5)/(vol+1e-12))* (1+surprise.clip(-.5,2))
f=f.rank(axis=1,pct=True).sub(.5,axis=0)
for h in [1,3,5,10]:
 fr=p.shift(-h).div(p)-1; xs=[]; ns=[]
 for dt in f.index.intersection(fr.index):
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: xs.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 x=pd.Series(xs).dropna(); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round(f.diff().abs().mean().mean(),4),'period',p.index.min(),p.index.max())
f.index=f.index.strftime('%Y-%m-%d');f.to_csv('scripts/miner_1_20330429_liquidity_reversal_signal.csv')
