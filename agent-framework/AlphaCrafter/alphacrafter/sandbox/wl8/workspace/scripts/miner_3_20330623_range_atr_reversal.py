import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
close=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-06-22']
high=pd.DataFrame({s:x.high.astype(float).replace(0,np.nan) for s,x in P.items()}).reindex(close.index); low=pd.DataFrame({s:x.low.astype(float).replace(0,np.nan) for s,x in P.items()}).reindex(close.index)
r=np.log(close).diff(); ret10=np.log(close/close.shift(10))
# range-normalized mean-reversion: reversal magnitude divided by 20d ATR-like log range volatility
rng=np.log(high/low).replace([np.inf,-np.inf],np.nan)
atr=rng.rolling(20,min_periods=10).mean()
base=(-ret10)/(atr+1e-12)
# robust cross-sectional clipping and percentile rank, activated by market-wide dispersion
base=base.clip(base.quantile(.10,axis=1),base.quantile(.90,axis=1),axis=0).rank(axis=1,pct=True).sub(.5)
disp=r.rolling(5,min_periods=5).std().mean(axis=1)
gate=disp.rolling(60,min_periods=30).rank(pct=True)
f=base.mul(gate,axis=0).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(close.shift(-10)/close).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=np.array([x[2] for x in rows])
print('candidate range_atr_dispersion_reversal dates',len(i),'avgN %.3f coverage %.5f'%(ns.mean(),ns.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean())); print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for h in [1,5,10,20]:
 a=[]; qq=np.log(close.shift(-h)/close)
 for d in f.index:
  z=pd.concat([f.loc[d],qq.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330623_range_atr_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330623_range_atr_reversal_ic.csv',index=False)
