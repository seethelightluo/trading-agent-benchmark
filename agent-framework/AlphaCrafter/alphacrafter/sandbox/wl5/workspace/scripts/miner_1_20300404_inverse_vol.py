import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2030-04-03')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index();P=P[P.index<=end]
R=P.pct_change(); vol=R.rolling(20).std(); sig=-vol
rows=[]
for h in [5,10,20]:
 a=[];cv=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));cv.append(len(z)/15)
 a=np.array(a);rows.append((h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),np.mean(cv)))
rank=sig.rank(axis=1,pct=True);print('cutoff',end.date(),'dates',len(P),'instruments',P.shape[1],'rows',rows,'turnover',float((rank.diff().abs().sum(axis=1)/15).mean()),'coverage',float(sig.notna().mean().mean()))
for name,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('recent','2029-01-01','2030-04-03')]:
 a=[]
 for i,t in enumerate(P.index[:-20]):
  if pd.Timestamp(lo)<=t<=pd.Timestamp(hi):
   z=pd.concat([sig.iloc[i],P.iloc[i+20]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(name,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
out=pd.DataFrame({'date':np.repeat(P.index,len(U)),'symbol':U*len(P),'signal':sig.to_numpy().ravel()});out.to_csv('scripts/miner_1_20300404_inverse_vol_signal.csv',index=False)
