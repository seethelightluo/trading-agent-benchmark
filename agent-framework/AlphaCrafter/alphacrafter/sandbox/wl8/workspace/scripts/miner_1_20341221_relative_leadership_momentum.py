import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=pd.to_numeric(x.close,errors='coerce')
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Relative leadership: lagged 20-session asset return versus cross-sectional median,
# normalized by asset 20d volatility. This isolates continuation of leaders/laggards.
ret20=p.pct_change(20); med=ret20.median(axis=1); vol=r.rolling(20,min_periods=15).std()
sig=((ret20.sub(med,axis=0))/vol).clip(-8,8).shift(1)
rows=[]
for dt in sig.index:
 y=p.shift(-10).loc[dt]/p.loc[dt]-1; z=pd.concat([sig.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); q=ic.ic
print('dates',len(ic),'avgN',round(ic.n.mean(),2),'coverage',round(sig.notna().mean().mean(),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rk=sig.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(to))
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,'ICIR',z.mean()/z.std(ddof=1),'IC',z.mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
out=sig.loc[ic.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20341221_relative_leadership_momentum_signal.csv',index=False);ic.reset_index().to_csv('scripts/miner_1_20341221_relative_leadership_momentum_ic.csv',index=False)
