import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1,sort=True).loc[:'2026-07-15']; r=np.log(p).diff(); m=r.median(axis=1)
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 a=pd.concat([r[s],m],axis=1).dropna();a.columns=['a','m']
 beta=a.a.rolling(60,min_periods=45).cov(a.m)/a.m.rolling(60,min_periods=45).var()
 # residual cumulative momentum, beta estimated with only trailing data
 resid=a.a-beta*a.m
 f.loc[a.index,s]=resid.rolling(20,min_periods=15).sum()
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));ds.append(dt)
 z=np.array(vals);d=pd.DatetimeIndex(ds);print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for y,mask in [('20-22',d<='2022-12-31'),('23-24',(d>='2023-01-01')&(d<='2024-12-31')),('25-26',d>='2025-01-01')]:
   zz=z[mask];print(y,len(zz),zz.mean(),zz.mean()/zz.std(ddof=1))
f.to_csv('scripts/miner_2_20260730_resid_mom_signal.csv');print('signal_artifact scripts/miner_2_20260730_resid_mom_signal.csv')
