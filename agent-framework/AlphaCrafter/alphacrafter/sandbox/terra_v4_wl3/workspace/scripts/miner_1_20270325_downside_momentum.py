import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={};cut=pd.Timestamp('2027-03-24')
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x[x.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index();R=P.pct_change();ret=P/P.shift(20)-1;down=R.where(R<0).rolling(20,min_periods=12).std();F=ret/down.replace(0,np.nan);F.to_csv('scripts/miner_1_20270325_downside_momentum_signal.csv')
for h in [1,5,10]:
 vals=[];ns=[];dates=[]
 for i in range(20,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v): vals.append(v);ns.append(len(z));dates.append(P.index[i])
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)))
 if h==1:
  ds=pd.Series(a,index=pd.DatetimeIndex(dates))
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=ds.loc[lo:hi];print('regime',lo,hi,'n',len(q),'IC',round(float(q.mean()),6))
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',P.index.min(),P.index.max())
