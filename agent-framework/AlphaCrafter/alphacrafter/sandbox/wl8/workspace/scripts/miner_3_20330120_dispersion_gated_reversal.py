import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff(); ret20=np.log(p/p.shift(20)); vol=lr.rolling(20,min_periods=15).std()
# Cross-asset dispersion gate: fade 10d moves more aggressively when market dispersion is elevated.
disp=lr.rolling(10,min_periods=8).std().mean(axis=1)
gate=(disp/disp.rolling(60,min_periods=30).median()).clip(0.5,2.0)
raw=(-np.log(p/p.shift(10))/vol.replace(0,np.nan))*gate.values[:,None]
f=raw.rank(axis=1,pct=True).sub(raw.rank(axis=1,pct=True).mean(axis=1),axis=0).rolling(2,min_periods=2).mean()
cut=pd.Timestamp('2033-01-20'); f=f.loc[:cut]; p=p.loc[:cut]
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows])
print('dates',len(i),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[]; q=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330120_dispersion_gated_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20330120_dispersion_gated_reversal_ic.csv',index=False)
