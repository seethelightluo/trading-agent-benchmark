import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close.astype(float).replace(0,np.nan)
 except FileNotFoundError: pass
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff()
# Trend consistency momentum: intermediate return weighted by fraction of positive daily moves,
# with cross-sectional daily demeaning and 3d smoothing to limit noise.
r30=np.log(p/p.shift(30)); consistency=(lr.gt(0).rolling(30,min_periods=20).mean()-0.5)*2
raw=r30*consistency
f=raw.sub(raw.mean(axis=1),axis=0).rolling(3,min_periods=2).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); print('dates',len(i),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[]; q=np.log(p.shift(-h)/p)
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20321209_consistency_momentum_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_3_20321209_consistency_momentum_ic.csv',index=False)
