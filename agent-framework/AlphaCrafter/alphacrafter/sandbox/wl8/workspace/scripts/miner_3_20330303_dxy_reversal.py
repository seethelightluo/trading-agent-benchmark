import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(p.index).ffill()
# DXY-strength-conditioned cross-asset reversal: contrarian 20d momentum is
# emphasized when dollar strength is above its trailing median, and attenuated
# during weak-dollar regimes. All inputs are lag-safe at the decision close.
trend=np.log(p/p.shift(20)); ds=np.log(dxy/dxy.shift(20)); strength=(ds-ds.rolling(120,min_periods=60).median()).clip(-.05,.05)
gate=(1+8*strength).clip(.5,1.5)
raw=-trend.mul(gate,axis=0); r=raw.rank(axis=1,pct=True); f=r.sub(r.mean(axis=1),axis=0).rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-03-02'); f=f.loc[:cut]; p=p.loc[:cut]
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=[x[2] for x in rows]
print('dates',len(i),'avgN',np.mean(ns),'coverage',np.mean(ns)/15); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',r.diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[];q=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330303_dxy_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330303_dxy_reversal_ic.csv',index=False)
