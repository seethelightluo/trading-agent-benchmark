import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index()
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(p.index).ffill()
r=np.log(p).diff(); dr=np.log(d).diff(); w=120
# lag-safe rolling beta and cumulative residual return over 20 sessions
cov=r.rolling(w,min_periods=60).cov(dr); var=dr.rolling(w,min_periods=60).var(); beta=cov.div(var,axis=0)
res=r-beta.mul(dr,axis=0); x=res.rolling(20,min_periods=15).sum()
rank=(-x).rank(axis=1,pct=True); base=rank.sub(rank.mean(axis=1),axis=0)
f=base.rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-04-27'); p=p.loc[:cut]; f=f.loc[:cut]
rows=[]
for d0 in f.index:
 z=pd.concat([f.loc[d0],np.log(p.shift(-10)/p).loc[d0]],axis=1).dropna()
 if len(z)>=8: rows.append((d0,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN %.3f coverage %.5f'%(ns.mean(),ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d0 in f.index:
  z=pd.concat([f.loc[d0],q.loc[d0]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330428_dxy_residual_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_1_20330428_dxy_residual_reversal_ic.csv',index=False)
