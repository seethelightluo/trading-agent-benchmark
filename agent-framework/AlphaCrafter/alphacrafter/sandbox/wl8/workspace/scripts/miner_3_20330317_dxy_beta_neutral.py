import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}
pd1=pd.DataFrame(P).sort_index(); lr=np.log(pd1).diff()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(pd1.index).ffill(); x=np.log(dxy).diff()
R=np.log(pd1/pd1.shift(40)); w=120
mx= x.rolling(w,min_periods=80).mean(); var=((x-mx)**2).rolling(w,min_periods=80).mean()
# rolling covariance, computed explicitly to avoid pandas DataFrame/Series cov alignment quirks
beta=lr.sub(mx,axis=0).mul(x-mx,axis=0).rolling(w,min_periods=80).mean().div(var,axis=0)
res=R-beta.mul(np.log(dxy/dxy.shift(40)),axis=0)
f=(-res.rank(axis=1,pct=True)).rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-03-16'); f=f.loc[:cut]; pd1=pd1.loc[:cut]
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(pd1.shift(-10)/pd1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([z[1] for z in rows],index=[z[0] for z in rows]); ns=np.array([z[2] for z in rows])
print('dates',len(i),'avgN %.3f'%ns.mean(),'coverage %.5f'%(ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 a=[]; q=np.log(pd1.shift(-h)/pd1)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330317_dxy_beta_neutral_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330317_dxy_beta_neutral_ic.csv',index=False)
