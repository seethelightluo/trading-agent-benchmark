import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=np.log(p).diff()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].reindex(r.index).ffill()
dr=np.log(dxy).diff()
# DXY-neutral 20d momentum: asset cumulative return minus rolling beta to DXY times DXY return
cov=r.rolling(60,min_periods=40).cov(dr); var=dr.rolling(60,min_periods=40).var(); beta=cov.divide(var,axis=0)
f=r.rolling(20,min_periods=15).sum()-beta*dr.rolling(20,min_periods=15).sum(axis=0) if False else r.rolling(20,min_periods=15).sum().sub(beta.mul(dr.rolling(20,min_periods=15).sum(),axis=0))
# score is known at t; forward returns after t
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:
   vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));dates.append(dt)
 z=np.array(vals); print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for name,ix in [('2020-22',(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31'))),('2023-24',(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31'))),('2025-26',(pd.Timestamp('2025-01-01'),pd.Timestamp('2026-07-15')))]:
 zz=np.array([v for v,d in zip(vals,dates) if ix[0]<=d<=ix[1]]); print(name,len(zz), 'IC',zz.mean() if len(zz) else np.nan,'ICIR',zz.mean()/zz.std(ddof=1) if len(zz)>1 else np.nan)
q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size,'period',p.index.min(),p.index.max())
print('corr existing momentum proxy',f.stack().corr(r.rolling(20,min_periods=15).sum().stack()))
