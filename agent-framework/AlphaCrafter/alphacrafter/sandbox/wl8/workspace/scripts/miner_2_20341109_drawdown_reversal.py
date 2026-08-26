import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=3900)
    except Exception: x=None
    if x is not None and len(x): D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
high=p.rolling(60,min_periods=45).max(); vol=r.rolling(20,min_periods=15).std(); trend=p.pct_change(120)
raw=-(1-p/high)/(vol*np.sqrt(20)); raw=raw.where(trend>0)
sig=raw.shift(1); fwd=p.shift(-10)/p-1
rows=[]; cov=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); cov.append(len(z)/len(U))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':].dropna(); vals=ic.ic
mean=vals.mean(); sd=vals.std(ddof=1); icir=mean/sd*np.sqrt(252)
ranks=sig.rank(axis=1,pct=True); turnovers=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(q)>=8: turnovers.append((q.iloc[:,1]-q.iloc[:,0]).abs().mean())
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(cov),'IC',mean,'ICIR',icir,'hit',np.mean(vals>0),'turnover',np.mean(turnovers))
for w in [365,750,1260]:
 q=vals.tail(w); print('recent',w,q.mean()/q.std(ddof=1)*np.sqrt(252))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr))
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20341109_drawdown_reversal_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_2_20341109_drawdown_reversal_ic.csv',index=False)
