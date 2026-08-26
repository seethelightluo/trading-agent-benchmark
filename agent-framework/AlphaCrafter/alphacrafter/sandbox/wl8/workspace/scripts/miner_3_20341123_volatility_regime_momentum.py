import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# Momentum acceleration: medium trend confirmed by short trend, scaled by volatility;
# lagged signal avoids look-ahead and forward horizon is 10 sessions.
raw=(0.55*p.pct_change(20)+0.45*p.pct_change(60))/(v*np.sqrt(20)); raw=raw.where(p.pct_change(120)>-0.05)
sig=raw.shift(1); f=p.shift(-10)/p-1; rows=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); cov.append(sig.loc[dt].notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); q=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(cov),'IC',q.mean(),'dailyICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rr=[]; rk=sig.rank(axis=1,pct=True)
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: rr.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(rr))
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,'ICIR',z.mean()/z.std(ddof=1))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
out=sig.loc[ic.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20341123_volatility_regime_momentum_signal.csv',index=False);ic.reset_index().to_csv('scripts/miner_3_20341123_volatility_regime_momentum_ic.csv',index=False)
