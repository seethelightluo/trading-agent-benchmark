import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); p=p.loc[:'2026-07-15']; r=np.log(p).diff()
f=-(r.rolling(10,min_periods=8).std()/r.rolling(60,min_periods=40).std())
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));ds.append(dt)
 z=np.array(vals); print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==1:
  for label,a0,b0 in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]:
   zz=np.array([v for v,d in zip(z,ds) if pd.Timestamp(a0)<=d<=pd.Timestamp(b0)]); print(label,len(zz),zz.mean(),zz.mean()/zz.std(ddof=1))
q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size,'period',p.index.min(),p.index.max())
print('corr riskadj20',f.stack().corr((r.rolling(20,min_periods=15).sum()/r.rolling(20,min_periods=15).std()).stack()))
