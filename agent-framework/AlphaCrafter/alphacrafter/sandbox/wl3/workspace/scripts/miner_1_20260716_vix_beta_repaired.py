import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].pct_change()
F={}; Y={}
for s,c in D.items():
 q=pd.concat([c.pct_change().rename('r'),v.rename('v')],axis=1).dropna()
 beta=q.r.rolling(60,min_periods=40).cov(q.v)/q.v.rolling(60,min_periods=40).var()
 F[s]=(-beta).rename(s)
 Y[s]=(c.shift(-1)/c-1).rename(s)
F=pd.DataFrame(F).loc[:end]; Y=pd.DataFrame(Y).loc[:end]
print('dates',len(F),'assets',len(U),'mean valid',F.notna().sum(axis=1).mean(),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 yh={}
 for s,c in D.items(): yh[s]=(c.shift(-h)/c-1)
 Yh=pd.DataFrame(yh).loc[:end]; vals=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Yh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('h',h,'N',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
R=[]
for dt in F.index:
 q=F.loc[dt].dropna(); R.append(q.rank(pct=True).reindex(U))
print('dates>=8',(F.notna().sum(axis=1)>=8).mean(),'turnover',pd.DataFrame(R,index=F.index).diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 vals=[]
 for dt in F.loc[lo:hi].index:
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('regime',lo,hi,'N',len(vals),'IC',np.mean(vals))
for nm,X in [('rev5',pd.DataFrame({s:-(c/c.shift(5)-1) for s,c in D.items()})),('mom20',pd.DataFrame({s:c/c.shift(20)-1 for s,c in D.items()})),('ram20',pd.DataFrame({s:(c/c.shift(20)-1)/c.pct_change().rolling(60).std() for s,c in D.items()}))]:
 q=pd.concat([F.stack(),X.loc[:end].stack()],axis=1).dropna(); print('corr',nm,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
