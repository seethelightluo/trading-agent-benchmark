import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
  P[s]=pd.to_numeric(x['close'],errors='coerce')
  if 'volume' in x: V[s]=pd.to_numeric(x['volume'],errors='coerce')
p=pd.DataFrame(P).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
v=pd.DataFrame(V).reindex(p.index).ffill()
# Contrarian five-day return, risk scaled, and confirmed by unusually high recent activity.
activity=(v.rolling(20,min_periods=12).mean()/v.rolling(60,min_periods=30).mean()).clip(.5,2.0)
sig=(-p.pct_change(5)/(vol*np.sqrt(5))*activity).shift(1)
rows=[]; cov=[]
for dt in sig.index:
 y=p.shift(-10).loc[dt]/p.loc[dt]-1
 z=pd.concat([sig.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 cov.append(sig.loc[dt].notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); q=ic.ic
print('dates',len(ic),'avgN',round(ic.n.mean(),3),'coverage',round(np.mean(cov),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
rk=sig.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',np.mean(to))
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,'ICIR',z.mean()/z.std(ddof=1),'IC',z.mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20341207_volume_confirmed_reversal_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_1_20341207_volume_confirmed_reversal_ic.csv',index=False)
