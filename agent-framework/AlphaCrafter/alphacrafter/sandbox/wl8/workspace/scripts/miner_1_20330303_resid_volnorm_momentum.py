import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff();
# Candidate: market-residualized intermediate momentum, volatility normalized and smoothed.
market=lr.mean(axis=1)
beta=lr.rolling(60,min_periods=40).cov(market).div(market.rolling(60,min_periods=40).var(),axis=0)
ret20=np.log(p/p.shift(20)); resid=ret20-beta.mul(np.log(p).diff(20).mean(axis=1),axis=0)
vol=lr.rolling(30,min_periods=20).std()*np.sqrt(30)
raw=resid/vol
rank=raw.rank(axis=1,pct=True); f=rank.sub(rank.mean(axis=1),axis=0).rolling(3,min_periods=3).mean()
cut=pd.Timestamp('2033-03-02'); f=f.loc[:cut]; p=p.loc[:cut]
rows=[]; q=np.log(p.shift(-10)/p)
for d in f.index:
 z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=[x[2] for x in rows]
print('candidate residualized_volnorm_momentum_20d')
print('dates',len(i),'avgN %.3f coverage %.4f'%(np.mean(ns),np.mean(ns)/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover %.6f'%rank.diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[]; qh=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],qh.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330303_resid_volnorm_momentum_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_1_20330303_resid_volnorm_momentum_ic.csv',index=False)
