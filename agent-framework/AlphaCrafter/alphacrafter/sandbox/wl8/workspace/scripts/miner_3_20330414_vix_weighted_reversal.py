import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index()
# Macro-conditioned medium-horizon reversal: stronger contrarian signal in elevated VIX regimes.
lr=np.log(p).diff(); ret=np.log(p/p.shift(20)); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(p.index).ffill()
vp=v.rolling(120,min_periods=60).rank(pct=True)
base=-ret.rank(axis=1,pct=True).sub((-ret.rank(axis=1,pct=True)).mean(axis=1),axis=0)
f=base.mul((0.5+vp).clip(0.5,1.5),axis=0).rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-04-13'); f=f.loc[:cut]; p=p.loc[:cut]
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN %.3f'%ns.mean(),'coverage %.5f'%(ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 a=[]; q=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330414_vix_weighted_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330414_vix_weighted_reversal_ic.csv',index=False)
