import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close.astype(float).replace(0,np.nan)
 except FileNotFoundError: pass
p=pd.DataFrame(D).sort_index(); lp=np.log(p); r=lp.diff(); r10=lp-lp.shift(10)
disp=r.T.rolling(5,min_periods=3).std().T.shift(1).mean(axis=1)
# Strict high-dispersion conditional reversal, threshold is lagged 75th percentile.
gate=(disp>disp.rolling(252,min_periods=60).quantile(.75)).astype(float)
vol=r.rolling(20,min_periods=15).std(); base=(-r10/vol.replace(0,np.nan)).rank(axis=1,pct=True)
f=base.sub(base.mean(axis=1),axis=0).mul(gate,axis=0).rolling(3,min_periods=3).mean()
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],(lp.shift(-10)-lp).loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); print('dates',len(i),'avgN %.3f coverage %.4f'%(np.mean([x[2] for x in rows]),np.mean([x[2] for x in rows])/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover %.6f'%base.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[]; q=lp.shift(-h)-lp
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330120_highdisp_reversal10_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i}).to_csv('scripts/miner_1_20330120_highdisp_reversal10_ic.csv',index=False)
