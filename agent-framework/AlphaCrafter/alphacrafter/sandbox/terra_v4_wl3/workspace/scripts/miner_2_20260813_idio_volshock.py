import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1,sort=True).loc[:'2026-07-15']; r=np.log(p).diff(); common=r.median(axis=1); res=r.sub(common,axis=0)
short=res.rolling(10,min_periods=8).std(); long=res.rolling(60,min_periods=40).std(); f=-(short/long)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); z=[]; ns=[]; ds=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'r':fw.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:z.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));ds.append(d)
 z=np.array(z);ds=pd.DatetimeIndex(ds); sd=z.std(ddof=1)
 print('h',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/sd,(z>0).mean()))
 if h==1: print('coverage %.4f turnover %.4f'%(f.notna().sum().sum()/f.size,f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
print('period',p.index.min(),p.index.max(),'assets',p.shape[1]);f.to_csv('scripts/miner_2_20260813_idio_volshock_signal.csv')
