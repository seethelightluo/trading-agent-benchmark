import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index().loc[:'2033-06-08']
r=np.log(p).diff(); ret10=np.log(p/p.shift(10))
down=r.where(r<0).rolling(20,min_periods=5).std(); base=(-ret10)/(down+1e-12)
base=base.clip(base.quantile(.10,axis=1),base.quantile(.90,axis=1),axis=0).rank(axis=1,pct=True).sub(.5)
disp=r.rolling(5,min_periods=5).std().mean(axis=1); gate=disp.rolling(60,min_periods=30).rank(pct=True)
f=base.mul(gate,axis=0).rolling(5,min_periods=5).mean(); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('candidate downside_dispersion_reversal dates',len(i),'avgN %.3f coverage %.5f'%(ns.mean(),ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 a=[]; qq=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],qq.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330609_downside_dispersion_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330609_downside_dispersion_reversal_ic.csv',index=False)
