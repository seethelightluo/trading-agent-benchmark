import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path): return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').close
p=pd.concat({s:load('../persistent/stock_data/'+s+'.csv') for s in U},axis=1,sort=True).loc[:'2026-07-15']
x=load('../persistent/index_data/DXY.csv').reindex(p.index).ffill(); r=np.log(p).diff(); q=np.log(x).diff()
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 a=pd.concat([r[s],q],axis=1).dropna(); a.columns=['a','q']
 cov=a.a.rolling(60,min_periods=45).cov(a.q); var=a.q.rolling(60,min_periods=45).var()
 f.loc[a.index,s]=-cov/var
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));ds.append(dt)
 z=np.array(vals); d=pd.DatetimeIndex(ds)
 print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for y,mask in [('20-22',d<='2022-12-31'),('23-24',(d>='2023-01-01')&(d<='2024-12-31')),('25-26',d>='2025-01-01')]:
   zz=z[mask];print(y,len(zz),zz.mean(),zz.mean()/zz.std(ddof=1))
f.to_csv('scripts/miner_2_20260730_dxy_beta_signal.csv');print('signal_artifact scripts/miner_2_20260730_dxy_beta_signal.csv')
