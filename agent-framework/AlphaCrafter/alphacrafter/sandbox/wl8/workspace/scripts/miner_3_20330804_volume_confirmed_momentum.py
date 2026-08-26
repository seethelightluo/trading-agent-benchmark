import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-08-03']
V=pd.DataFrame({s:x.volume.astype(float).replace(0,np.nan) for s,x in P.items()}).reindex(C.index)
r=np.log(C).diff()
# Volume-confirmed intermediate momentum: 20d return, strengthened by relative 5d-vs-60d volume activity.
ret=np.log(C/C.shift(20))
vr=(V.rolling(5,min_periods=3).mean()/V.rolling(60,min_periods=30).mean()).replace([np.inf,-np.inf],np.nan)
# Cross-sectional percentile volume confirmation, centered, with modest 0.5 loading.
qret=ret.rank(axis=1,pct=True)-.5
qvol=vr.rank(axis=1,pct=True)-.5
f=(qret*(1+0.5*qvol)).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(C.shift(-10)/C).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); n=pd.Series([x[2] for x in rows],index=i.index)
print('dates',len(i),'avgN',n.mean(),'coverage',n.mean()/15)
print('IC',i.mean(),'ICIR',i.mean()/i.std(ddof=1),'hit',(i>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for w in [365,750,1260]:
 q=i.tail(w);print('recent',w,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 a=[]; q=np.log(C.shift(-h)/C)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330804_volume_confirmed_momentum_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20330804_volume_confirmed_momentum_ic.csv')
