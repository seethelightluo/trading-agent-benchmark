import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(P,axis=1,sort=True).loc[:'2026-07-15']; r=np.log(p).diff()
neg=r.where(r<0,0.0); dv=np.sqrt((neg**2).rolling(30,min_periods=20).mean()); f=r.rolling(10,min_periods=10).sum()/dv.replace(0,np.nan)
vals=[];ns=[];dates=[]
for dt in f.index:
 a=pd.DataFrame({'f':f.loc[dt],'r':p.pct_change(1).shift(-1).loc[dt]}).dropna()
 if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));dates.append(dt)
z=np.array(vals);d=pd.DatetimeIndex(dates);print('candidate tail_adjusted_10d_momentum');print('period',p.index.min().date(),p.index.max().date(),'assets',p.shape[1]);print('dates',len(z),'avg_names',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean());print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,mask in [('2020-22',d<='2022-12-31'),('2023-24',(d>='2023-01-01')&(d<='2024-12-31')),('2025-26',d>='2025-01-01')]:
 q=z[mask];print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_3_20260910_tail_adjusted_signal.csv');print('signal_artifact scripts/miner_3_20260910_tail_adjusted_signal.csv')
