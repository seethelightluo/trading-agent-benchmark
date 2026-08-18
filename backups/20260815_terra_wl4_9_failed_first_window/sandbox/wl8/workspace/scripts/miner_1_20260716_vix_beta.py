import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index() for s in U}
C=pd.DataFrame({s:d.close for s,d in D.items()}); r=C.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.pct_change().reindex(C.index)
vr=v.rolling(60,min_periods=40).var(); cov=r.apply(lambda x:x.rolling(60,min_periods=40).cov(v)); F=-cov.divide(vr,axis=0).loc[:end]
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for dt in F.index:
  x=[];y=[]
  for s in U:
   if pd.isna(F.loc[dt,s]) or dt not in D[s].index:continue
   i=D[s].index.get_loc(dt);j=i+h
   if j<len(D[s]) and pd.notna(D[s].iloc[j].close):x.append(F.loc[dt,s]);y.append(D[s].iloc[j].close/D[s].iloc[i].close-1)
  if len(x)>=8 and np.nanstd(x)>0: vals.append(pd.Series(x).corr(pd.Series(y),method='spearman'));ns.append(len(x))
 a=np.array(vals);print('h',h,'N',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',(F.notna().sum(axis=1)>=8).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=[]
 for dt in F.loc[lo:hi].index:
  x=[];y=[]
  for s in U:
   if pd.notna(F.loc[dt,s]) and dt in D[s].index:
    i=D[s].index.get_loc(dt)
    if i+1<len(D[s]):x.append(F.loc[dt,s]);y.append(D[s].iloc[i+1].close/D[s].iloc[i].close-1)
  if len(x)>=8:z.append(pd.Series(x).corr(pd.Series(y),method='spearman'))
 print('regime',lo,hi,len(z),np.mean(z))
for nm,X in [('rev5',-(C/C.shift(5)-1)),('mom20',C/C.shift(20)-1),('ram20',(C/C.shift(20)-1)/r.rolling(60).std())]:
 z=pd.concat([F.stack(),X.loc[:end].stack()],axis=1).dropna();print('corr',nm,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
