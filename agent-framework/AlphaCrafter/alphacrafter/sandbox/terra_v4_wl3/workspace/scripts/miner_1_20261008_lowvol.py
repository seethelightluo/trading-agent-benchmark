import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U},axis=1,sort=True).loc[:'2026-10-07']
r=np.log(p).diff(); f=-r.rolling(20,min_periods=15).std(); fw=p.pct_change().shift(-1)
vals=[];ns=[];dates=[]
for dt in f.index:
 a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
 if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));dates.append(dt)
z=np.array(vals);d=pd.DatetimeIndex(dates)
print(f'dates={len(z)} meanN={np.mean(ns):.2f} assets={len(U)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
print('coverage=',f.notna().sum().sum()/f.size,'turnover=',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,mask in [('2020-22',d<='2022-12-31'),('2023-24',(d>='2023-01-01')&(d<='2024-12-31')),('2025-26',d>='2025-01-01')]:
 q=z[mask];print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20261008_lowvol_signal.csv');print('signal_artifact scripts/miner_1_20261008_lowvol_signal.csv')
