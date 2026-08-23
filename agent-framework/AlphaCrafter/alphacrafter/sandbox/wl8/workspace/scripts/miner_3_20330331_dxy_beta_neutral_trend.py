import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index()
lr=np.log(p).diff(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(p.index).ffill(); x=np.log(dxy).diff()
w=120; mx=x.rolling(w,min_periods=80).mean(); var=((x-mx)**2).rolling(w,min_periods=80).mean()
beta=lr.sub(mx,axis=0).mul(x-mx,axis=0).rolling(w,min_periods=80).mean().div(var,axis=0)
# DXY-neutralized intermediate trend: residual 40d return, smoothed cross-sectional rank.
res=np.log(p/p.shift(40))-beta.mul(np.log(dxy/dxy.shift(40)),axis=0)
r=res.rank(axis=1,pct=True); f=r.sub(r.mean(axis=1),axis=0).rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-03-30'); f=f.loc[:cut]; p=p.loc[:cut]
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([a[1] for a in rows],index=[a[0] for a in rows]); ns=np.array([a[2] for a in rows])
print('dates',len(i),'avgN %.3f'%ns.mean(),'coverage %.5f'%(ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%r.diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 a=[];q=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330331_dxy_beta_neutral_trend_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330331_dxy_beta_neutral_trend_ic.csv',index=False)
