import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index()
cut=pd.Timestamp('2033-05-11'); p=p.loc[:cut]
logr=np.log(p).diff()
# Medium trend with recent-shock reversal: favor persistent 20d strength, but fade the last 3d shock.
# Normalize each asset by trailing 20d realized volatility, then cross-sectional rank and smooth.
vol=logr.rolling(20,min_periods=15).std()*np.sqrt(20)
medium=np.log(p/p.shift(20)); shock=np.log(p/p.shift(3))
raw=medium/(vol+1e-12)-0.50*shock/(vol+1e-12)
# Winsorize cross-section and rank to make scale comparable across assets
raw=raw.clip(lower=raw.quantile(.10,axis=1),upper=raw.quantile(.90,axis=1),axis=0)
f=raw.rank(axis=1,pct=True).sub(.5).rolling(3,min_periods=3).mean()
fr=np.log(p.shift(-10)/p)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('candidate volnorm_persistence_reversal dates',len(i),'avgN %.3f'%ns.mean(),'coverage %.5f'%(ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330512_volnorm_persistence_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330512_volnorm_persistence_reversal_ic.csv',index=False)
